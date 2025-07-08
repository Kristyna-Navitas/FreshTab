import argparse
import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import re
import torch

from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from transformers import BartForSequenceClassification, TapexTokenizer, DataCollatorWithPadding, \
    TapasForSequenceClassification, TapasTokenizer

from logzero import logger

from evaluation.metrics import run_self_bleu, get_avg_length, get_unique_tokens, get_shannon_entropy, get_msttr
from utils import save_json


def process_dataset(dataset, tokenizer, length):
    # from https://github.com/yale-nlp/LLM-T2T/blob/main/src/open_src_model_T2T_generation.py
    def process_example(example):
        table_df = pd.read_csv(StringIO(example['table_csv'])).astype(str)
        inp = tokenizer(
            table_df,
            example['prediction'].rstrip('. '),  # for compatibility with tabfact
            max_length=length,
            truncation=True
        )
        return inp

    inp_columns = ["label", "input_ids", "attention_mask", "token_type_ids"]
    res_dataset = dataset.map(process_example, batched=False)
    extra_columns = [col for col in res_dataset.features.keys() if col not in inp_columns]
    res_dataset = res_dataset.remove_columns(extra_columns)

    return res_dataset


def evaluate_dataset(dataset, batch, model, tokenizer, collator, length):
    all_nli_preds = []

    proc_dataset = process_dataset(dataset, tokenizer, length)
    dataloader = DataLoader(proc_dataset, batch_size=batch, collate_fn=collator)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            nli_preds = model(**batch).logits.cpu()
            nli_preds = np.argmax(nli_preds, axis=1).tolist()
            all_nli_preds.extend(nli_preds)

    return all_nli_preds


def load_predictions(preds):
    predictions_dict, csv_ids_dict = {}, {}
    with open(preds, mode='r', newline='') as file:
        reader = csv.reader(file)
        count = 0  # because we have more predictions for one table
        for row in reader:
            row = [value if value else 'nan' for value in row]
            csv_ids_dict[count] = row[0]
            if row[1]:
                predictions_dict[count] = row[6]
            else:
                predictions_dict[count] = 'wrong'
            count += 1

    return predictions_dict, csv_ids_dict


def get_tables(csv_dict, tables) -> dict:
    tables_dict = {}
    for key, value in csv_dict.items():
        tables_dict[key] = pd.read_csv(tables / value, sep='#')

    return tables_dict


def create_dataset(csv_dict, predictions, tables):
    items = []
    for key, value in predictions.items():
        items.append({
            'id': csv_dict[key]+'_'+str(key),  # because we have more predictions for one table
            'table_csv': tables[key].to_csv(index=False),
            'prediction': value
        })
    return Dataset.from_list(items)


def evaluate(args):
    name_stem = str(Path(args.preds).stem)
    # loading the predictions:
    if args.only_stats:
        results_df = pd.read_csv(args.preds, header=0)
        output_path = f"outputs/{name_stem}.csv"
        get_detailed_stats(results_df, output_path)

        return

    predictions_dict, csv_dict = load_predictions(args.preds)
    tables_dict = get_tables(csv_dict, args.tables)
    dataset = create_dataset(csv_dict, predictions_dict, tables_dict)

    # TAPEX
    model_ex = BartForSequenceClassification.from_pretrained('microsoft/tapex-large-finetuned-tabfact')
    tokenizer_ex = TapexTokenizer.from_pretrained('microsoft/tapex-large-finetuned-tabfact', add_prefix_space=True)
    collator_ex = DataCollatorWithPadding(tokenizer_ex)
    # TAPAS
    model_as = TapasForSequenceClassification.from_pretrained('google/tapas-large-finetuned-tabfact')
    tokenizer_as = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-tabfact')
    collator_as = DataCollatorWithPadding(tokenizer_as)

    # predictions
    predictions_tapex = evaluate_dataset(dataset, args.evaluate_batch, model_ex, tokenizer_ex, collator_ex, 1024)
    tapex_score = count_percentage(predictions_tapex)
    print('TAPEX score:', tapex_score)
    predictions_tapas = evaluate_dataset(dataset, args.evaluate_batch, model_as, tokenizer_as, collator_as, 512)
    tapas_score = count_percentage(predictions_tapas)
    print('TAPAS score:', tapas_score)

    results = pd.read_csv(args.preds, names=['csv_ids', 'domain', 'label', 'prediction'])
    results['TAPEX'] = predictions_tapex
    results['TAPAS'] = predictions_tapas

    output_path = f"outputs/{name_stem}_evaluated.csv"
    results.to_csv(output_path, index=True)
    logger.info(f'Predictions saved to {output_path}.')

    # saving detailed resutls to filename +_results.csv
    get_detailed_stats(results, output_path)

    return output_path, tapex_score, tapas_score


def get_label_following_simple(sentence: str) -> list[str]:
    """
    Identifies potential logical operations present in a sentence based on keywords.
    Args: sentence: The input sentence (string).
    Returns: A list of matching operation types.
    """
    sentence_lower = sentence.lower()
    matches = []

    ordinal_regex = re.compile(r"\b\d+(?:st|nd|rd|th)\b")
    count_regex = re.compile(r"there (?:are|were|have been)\s+\d+\b")
    super_regex = re.compile(r"\bthe\s+.*?\w+est\b")
    written_nums = "(?:one|two|three|four|five|six|seven|eight|nine|ten|)"
    written_num_regex = re.compile(rf"there (?:are|were)\s+{written_nums}\b")
    operations = {
        'aggregation': ['average', 'total', 'sum of', 'count'],
        'negation': ['not ', ' not',  'never', 'no ', ' no', 'none'],
        'superlative': ['most', 'least', 'worst', super_regex],
        'count': ['number of', 'total of', count_regex, written_num_regex],
        'comparative': ['higher', 'lower', 'more', 'less', 'greater', 'smaller', 'older',
                        'younger', 'longer', 'shorter', 'same', 'fewer', 'compared to'],
        'ordinal': ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth',
                        'ninth', 'tenth', 'last', ordinal_regex],
        'unique': ['different', 'unique', 'only', 'distinct'],
        'all': ['all', 'every', 'each', 'none of'],
        'simple': []
    }
    found_matches = False
    for op_type, keywords in operations.items():
        if op_type == 'simple':  # Skip for initial keyword matching
            continue
        for keyword in keywords:
            if isinstance(keyword, str) and keyword in sentence_lower:
                matches.append(op_type)
                found_matches = True
                break
            elif isinstance(keyword, re.Pattern) and keyword.search(sentence_lower):
                matches.append(op_type)
                found_matches = True
                break

    return matches


def get_detailed_stats(results: pd.DataFrame, filename: str) -> None:
    results.rename(columns={'query': 'sql_query'}, inplace=True)
    results_empty = results[(results['prediction'].isnull()) | (results['prediction'].str.len() < 5) | (results['prediction'] == 'DUMMY')]
    results_cleaned = results[(results['prediction'].notnull()) & (results['prediction'].str.len() >= 5) & (results['prediction'] != 'DUMMY')]
    num_results = len(results)
    empty_tables = []
    for table in results.csv_ids.unique().tolist():
        all_empty = results[results.csv_ids == table].prediction.isnull().all() + len(results[results['prediction'] == 'DUMMY'])
        if all_empty:
            empty_tables.append(table)
    results_dict = {
        'File name': filename,
        'all predictions': len(results),
        'empty predictions': len(results_empty),
        'not empty predictions': len(results_cleaned),
        'all empty table': len(empty_tables),
    }
    if num_results != 0:
        results_dict['percent empty'] = len(results_empty) / num_results
        results_dict['percent not empty'] = len(results_cleaned) / num_results

    # statistics
    ids = [x for x in results.csv_ids]
    list_of_sentences = []
    for table in ids:
        list_of_sentences.append(results[results.csv_ids == table].prediction.tolist())
    results_dict['statistics'] = {
        'BLEU': run_self_bleu(list_of_sentences),
        'avg_length': get_avg_length(list_of_sentences),
        'unique_tokens': get_unique_tokens(list_of_sentences),
        'entropy': get_shannon_entropy(list_of_sentences),
        'msttr': get_msttr(list_of_sentences)
    }

    # metrics
    lower_tapas, upper_tapas = proportion_confint(count=results.TAPAS.sum(), nobs=num_results, alpha=0.05,  # For a 95% CI
                                                  method='wilson')
    lower_tapex, upper_tapex = proportion_confint(count=results.TAPEX.sum(), nobs=num_results, alpha=0.05,  # For a 95% CI
                                                  method='wilson')
    results_dict['metrics'] = {
        'TAPAS': {
            'for_whole': count_percentage(results.TAPAS.tolist()),
            'not_empty': count_percentage(results_cleaned.TAPAS.tolist()),
            'empty': count_percentage(results_empty.TAPAS.tolist()),
            'upper_bound': lower_tapas,
            'lower_bound': upper_tapas,
        },
        'TAPEX': {
            'for_whole': count_percentage(results.TAPEX.tolist()),
            'not_empty': count_percentage(results_cleaned.TAPEX.tolist()),
            'empty': count_percentage(results_empty.TAPEX.tolist()),
            'upper_bound': lower_tapex,
            'lower_bound': upper_tapex,
        },
    }

    # for different logical labels
    labels = results.label.unique().tolist()
    total_correct_labels = 0
    labels_dict, labelsacc_dict = {}, {}
    for label in labels:
        all_results = results[results.label == label]
        clean_results = results_cleaned[results_cleaned.label == label]
        calculated = analysis_for_group(filename, label, all_results, clean_results)
        labels_dict.update(calculated)
        correct_labels = 0
        for sentence in all_results.prediction.tolist():
            found_labels = get_label_following_simple(sentence)
            if label == 'simple':
                if not found_labels:
                    correct_labels += 1
            elif label in found_labels:
                    correct_labels += 1
        labelsacc_dict[label+' count'] = correct_labels
        labelsacc_dict[label+' percentage'] = correct_labels/len(all_results)
        total_correct_labels += correct_labels
    labelsacc_dict['everything count'] = total_correct_labels
    labelsacc_dict['everything percentage'] = total_correct_labels/num_results
    results_dict['operations_metrics'] = labels_dict

    # for different domains
    results_dict['operations_accuracy'] = labelsacc_dict
    domains = results.domain.unique().tolist()
    dm_dict = {}
    for domain in domains:
        all_results = results[results.domain == domain]
        clean_results = results_cleaned[results_cleaned.domain == domain]
        calculated = analysis_for_group(filename, domain, all_results, clean_results)
        results_dict.update(calculated)
    results_dict['domains'] = dm_dict

    # combined domain and label
    dmlb_dict = {}
    for label in labels:
        for domain in domains:
            all_results = results[(results.domain == domain) & (results.label == label)]
            clean_results = results_cleaned[(results_cleaned.domain == domain) & (results.label == label)]
            calculated = analysis_for_group(filename, domain+' '+label, all_results, clean_results)
            dmlb_dict.update(calculated)
    results_dict['domain_label'] = dmlb_dict

    save_json(results_dict, f"generation/{filename}_results.json")


def analysis_for_group(filename, group_name, full_df, cleaned_df):
    length = len(full_df)
    lower_tapas, upper_tapas = proportion_confint(count=full_df.TAPAS.sum(), nobs=length, alpha=0.05,  # For a 95% CI
                                                  method='wilson')
    whole_tapex = count_percentage(full_df.TAPEX.tolist())
    lower_tapex, upper_tapex = proportion_confint(count=full_df.TAPEX.sum(), nobs=length, alpha=0.05,  # For a 95% CI
                                                  method='wilson')
    results_dict = {
        f'{group_name} count': length,
        f'{group_name} not empty count': len(cleaned_df),
    }
    if len(full_df) != 0:
        results_dict[f'{group_name} not empty percentage'] = len(cleaned_df) / len(full_df)

    results_dict[f'{group_name} metrics'] = {
        'TAPAS': {
            'whole': count_percentage(full_df.TAPAS.tolist()),
            'not_empty': count_percentage(cleaned_df.TAPAS.tolist()),
            'lower_bound': lower_tapas,
            'upper_bound': upper_tapas,
        },
        'TAPEX': {
            'whole': count_percentage(full_df.TAPEX.tolist()),
            'not_empty': count_percentage(cleaned_df.TAPEX.tolist()),
            'lower_bound': lower_tapex,
            'upper_bound': upper_tapex,
        },
    }

    return results_dict


def count_percentage(prediction):
    if len(prediction) != 0:
        return sum(prediction)/len(prediction)
    else:
        return 0


def main():
    argparser = argparse.ArgumentParser(
        description='Script for evaluating claims about tables from LogicNLG using TAPAX and TAPEX metrics.')

    argparser.add_argument('--preds', type=Path,
                                    help='Input dataset with zero first as ids of the tables, third column predictions.'
                                         'The output will be saved with added "_evaluated" to the name.')
    argparser.add_argument('--tables', type=Path,
                                    help='Path to the folder with all the csv tables.')
    argparser.add_argument('--only_stats', action="store_true",
                           help='Only counts stats, does not evaluate TAPEX and TAPAS, must already be there.')
    argparser.add_argument('--evaluate_batch', type=int, default=32,
                                    help='Well, batch size...')
    argparser.set_defaults(method=evaluate)

    args = argparser.parse_args()
    args.method(args)


if __name__ == '__main__':
    main()
