import argparse
import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from transformers import BartForSequenceClassification, TapexTokenizer, DataCollatorWithPadding, \
    TapasForSequenceClassification, TapasTokenizer

from logzero import logger

from utils import save_json


def process_dataset(dataset, tokenizer, length):
    # from https://github.com/yale-nlp/LLM-T2T/blob/main/src/open_src_model_T2T_generation.py
    def process_example(example):
        table_df = pd.read_csv(StringIO(example['table_csv'].replace('\\n', '\n').strip()), sep='#').astype(str)
        print(len(tokenizer.tokenize(' '.join(table_df.values.flatten()))))
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
                predictions_dict[count] = row[3]
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
    # loading the predictions:
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

    name_stem = str(Path(args.preds).stem)
    output_path = f"generation/outputs/{name_stem}_evaluated.csv"
    results.to_csv(output_path, index=True)
    logger.info(f'Predictions saved to {output_path}.')

    # saving detailed resutls to filename +_results.csv
    get_detailed_stats(results, output_path)

    return output_path, tapex_score, tapas_score


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

    # metrics
    results_dict['metrics'] = {
        'TAPAS': {
            'for_whole': count_percentage(results.TAPAS.tolist()),
            'not_empty': count_percentage(results_cleaned.TAPAS.tolist()),
            'empty': count_percentage(results_empty.TAPAS.tolist()),
        },
        'TAPEX': {
            'for_whole': count_percentage(results.TAPEX.tolist()),
            'not_empty': count_percentage(results_cleaned.TAPEX.tolist()),
            'empty': count_percentage(results_empty.TAPEX.tolist()),
        },
    }

    # for different logical labels
    labels = results.label.unique().tolist()
    for label in labels:
        all_results = results[results.domain == label]
        clean_results = results_cleaned[results_cleaned.domain == label]
        calculated = analysis_for_group(filename, label, all_results, clean_results)
        results_dict.update(calculated)
    # for different domains
    domains = results.domain.unique().tolist()
    for domain in domains:
        all_results = results[results.domain == domain]
        clean_results = results_cleaned[results_cleaned.domain == domain]
        calculated = analysis_for_group(filename, domain, all_results, clean_results)
        results_dict.update(calculated)
    # combined domain and label
    for label in labels:
        for domain in domains:
            all_results = results[(results.domain == domain) & (results.label == label)]
            clean_results = results_cleaned[(results_cleaned.domain == domain) & (results_cleaned.label == label)]
            calculated = analysis_for_group(filename, domain+' '+label, all_results, clean_results)
            results_dict.update(calculated)

    save_json(results_dict, f"{filename}_results.json")

def analysis_for_group(filename, group_name, full_df, cleaned_df):
    results_dict = {
        f'{group_name} count': len(full_df),
        f'{group_name} not empty count': len(cleaned_df),
    }
    if len(full_df) != 0:
        results_dict[f'{group_name} not empty percentage'] = len(cleaned_df) / len(full_df)

    results_dict[f'{group_name} metrics'] = {
        'TAPAS': {
            'whole': count_percentage(full_df.TAPAS.tolist()),
            'not_empty': count_percentage(cleaned_df.TAPAS.tolist()),
        },
        'TAPEX': {
            'whole': count_percentage(full_df.TAPEX.tolist()),
            'not_empty': count_percentage(cleaned_df.TAPEX.tolist()),
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
    argparser.add_argument('--evaluate_batch', type=int, default=32,
                                    help='Well, batch size...')
    argparser.set_defaults(method=evaluate)

    args = argparser.parse_args()
    args.method(args)


if __name__ == '__main__':
    main()
