import argparse
from pathlib import Path

import pandas as pd

from generation.helpers import OllamaClient, csv_writer
from generation.prompts import PROMPT_FACTUALITY
from utils import read_json


def process(args):
    dataset = read_json(args.data)
    predictions = pd.read_csv(args.preds, index_col=0)
    output_claims_path = Path('evaluation/llm_outputs/' + args.output)
    ollama_client = OllamaClient(
        host=f'http://{args.host}:{args.port}', model=args.model,
        decoding_options={'seed': args.seed, 'temperature': 0.7, 'min_p': args.min_p})

    # generate annotations
    with csv_writer(str(output_claims_path) + '.csv') as out_writer:
        for idcko, example in dataset.items():
            claims = predictions[predictions.csv_ids == example['csv_id']].prediction.tolist()
            for claim in claims:
                prompt = PROMPT_FACTUALITY.format(
                    title=example['title'],
                    table=example['table_text'],
                    claim=claim
                )
            response, prompt_str = ollama_client("", prompt)
            print(response)
            print([example['csv_id']])
            out_writer.writerow([example['csv_id']], claim, list(claim))


def main():
    argparser = argparse.ArgumentParser(
        description='Script for predicting claims about tables from FresthTab or LoTNLG dataset.')
    argparser.add_argument('--data', help='Input dataset in processed LogicNLG format in json.'
                                            'Needs csv_ids, title and table.')
    argparser.add_argument('--preds', help='CSV with predictions')
    argparser.add_argument('--output', type=str, default=None,
                           help='Name of the output directory to save it into.')
    argparser.add_argument('--host', default='localhost', help='For ollama server.')
    argparser.add_argument('--port', type=int, default=11434, help='For ollama server.')
    argparser.add_argument('--model', default="llama3.3:70b-instruct-q8_0",
                           help='Name of LLM for predictions for ollama.')
    argparser.add_argument('--min_p', type=float, default=0.02,
                           help='If none 0.0 will be used 0.02 is recommended value')
    argparser.add_argument('--seed', type=int, default=42)

    args = argparser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
