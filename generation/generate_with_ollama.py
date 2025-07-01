import argparse
import json
import logging
from pathlib import Path
from typing import List

from logzero import logger
from pydantic import BaseModel, Field, ValidationError

from .helpers import csv_writer, dump_to_LogicNLG_output, OllamaClientStructuredOutput, \
    dataframe_table_from_dataset, format_dataframe
from .prompts import SYSTEM_MSG, PROMPT_CHOICE, PROMPT_WITH_LABEL, explanation
from utils import read_json

EXPECTED_LABELS = ['superlative', 'aggregation', 'negation', 'comparative', 'unique', 'all', 'count', 'ordinal', 'simple']


class Experiment:
    """Class for generating claims directly from tables using an LLM."""
    def __init__(self, args):
        self.args = args
        self.llm = OllamaClientStructuredOutput(
            InsightsLabelsCoT, host=f'http://{args.host}:{args.port}', model=args.model, 
            decoding_options={'seed': args.seed, 'temperature': 0.7, 'min_p': args.min_p})

    def __call__(self, example):
        """Generate claims for a given example."""
        datatable = dataframe_table_from_dataset(example['table'])
        domain = example.get('category', 'none')
        if domain != 'none':
            domain = domain.split('|')[0]
        logger.info(f"{example['csv_id']=} {example['title']=}")
        labels = example.get('logical_labels', 'simple')
        labels = [l if l != 'none' else 'simple' for l in labels]  # the thinking model is overthinking

        if self.args.exp == 'choice':
            final_claims, labels, errors, _prompt = generate_structured_claims(
                self.args, self.llm, example['title'], datatable, logical_label=labels)
        elif self.args.exp == 'direct_cot':
            final_claims, errors = [], []
            for label in labels:
                claim, lbl, error, _prompt = generate_structured_claims(
                    self.args, self.llm, example['title'], datatable, logical_label=label)
                if claim:
                    if lbl[0] == label:
                        final_claims.append(claim[0])
                    else:
                        final_claims.append('DUMMY')
                        errors.append('label does not match')
                errors.append(error)
        logger.debug(f"{len(final_claims)=} {errors=}")

        claims = []
        for label, claim in zip(labels, final_claims):
            # generate sentence from the goal and retrieved data; not whole table text, just header
            claims.append([domain, label, claim])
        for i in range(5 - len(claims)):  # to have 5 claims in the output always
            DUMMY_LABEL = "surface"  # existing / valid label which we will use for dummy section
            claims.append([domain] + [DUMMY_LABEL] + ["DUMMY"])
        return claims

    def final(self):
        """Cleanup method called after all examples are processed."""
        pass


def generate_structured_claims(args, llm, title, table, logical_label):
    if args.exp == "direct_cot":
        prompt = PROMPT_WITH_LABEL.format(
            ideas_schema=InsightsLabelsCoT.model_json_schema(),
            title=title,
            table_columns=table.columns.tolist(),
            table=format_dataframe(table),
            logical_operation=logical_label,
            logical_operation_explanation=explanation[logical_label],
        )
    elif args.exp == "choice":
        prompt = PROMPT_CHOICE.format(
            ideas_schema=InsightsLabelsCoT.model_json_schema(),
            title=title,
            table_columns=table.columns.tolist(),
            table=format_dataframe(table),
            num_ideas=args.num_insights,
        )
    response, prompt_str = llm(SYSTEM_MSG, prompt)
    error, ideas = '', []
    try:
        ideas_labels_model = InsightsLabelsCoT.model_validate_json(response)
        ideas, labels, error = ideas_labels_model.insights, ideas_labels_model.labels, ''
        error_labels = [l for l in labels if l not in EXPECTED_LABELS]
        if error_labels:
            error = f": The following idea labels {error_labels} are not among the expected labels {EXPECTED_LABELS}"
        # HACK we do not retry for this we just filter it out and log it
        valid_ideas_labels = [(i, l) for i, l in zip(ideas, labels) if l in EXPECTED_LABELS]
        ideas = [i for i, l in valid_ideas_labels]
        labels = [l for i, l in valid_ideas_labels]
    except ValidationError as e:
        ideas, labels, error = [], [], str(e)
        logger.error(f"Error while parsing ideas: {e}\n{prompt_str=}\n{response=}")

    return ideas, labels, error, prompt_str


class InsightAndLabel(BaseModel):
    operation: str = Field(description=f"Logical operation applied. Must be exactly one of: {EXPECTED_LABELS}.")
    insight: str = Field(description="Your data insight idea.")


class InsightsLabelsCoT(BaseModel):
    thoughts: str = Field(description="Your step-by-step thoughts which data insights ideas are worth generating.")
    insight_label_pairs: List[InsightAndLabel] = Field(description="The final answer.")

    @property
    def labels(self):
        return [t.operation for t in self.insight_label_pairs]

    @property
    def insights(self):
        return [t.insight for t in self.insight_label_pairs]


def process(args):
    generate_table_claims = Experiment(args)

    dataset = read_json(args.input)
    output_claims_path = Path('generation/outputs/' + args.output)
    logger.info(f"{output_claims_path=}")

    # generate claims
    with csv_writer(str(output_claims_path) + '.csv') as out_writer:
        for idx, (csv_id, example) in enumerate(dataset.items()):
            logger.debug(f'Generating claims for {idx}-th table {csv_id}')
            claims = generate_table_claims(example)
            for claim in claims:
                print([example['csv_id']] + claim)
                out_writer.writerow([example['csv_id']] + claim)

            logger.info(f'Generated claims for {idx+1} table {csv_id}. {output_claims_path=}')

    generate_table_claims.final()
    dump_to_LogicNLG_output(str(output_claims_path) + '.csv', str(output_claims_path) + ".json")

    # save pretty printed arguments to json file
    with open(Path('generation/outputs') / f"{args.output}.args.json", "wt") as f:
        json.dump(vars(args), f, indent=2)

    logger.info('All done.')


def main():
    argparser = argparse.ArgumentParser(
        description='Script for predicting claims about tables from FresthTab or LoTNLG dataset.')
    argparser.add_argument('--exp', choices=["direct_cot", "choice"],
                           help='Which experiment to run. Allowed values are "direct_cot" or "choice".')
    argparser.add_argument('--input', help='Input dataset in processed LogicNLG format in json.'
                                           'Needs csv_ids, title, logical labels and table.')
    argparser.add_argument('--output', type=str, default=None,
                           help='Name of the output directory to save it into.')
    argparser.add_argument('--host', default='localhost', help='For ollama server.')
    argparser.add_argument('--port', type=int, default=11434, help='For ollama server.')
    argparser.add_argument('--model', default="llama3.3:70b-instruct-q8_0",
                           help='Name of LLM for predictions for ollama.')
    argparser.add_argument('--num_insights', type=int, default=5, help='Number of insights to generate.')
    argparser.add_argument('--log_level', default='INFO', help='Logging level.')
    argparser.add_argument('--min_p', type=float, default=0.02,
                           help='If none 0.0 will be used 0.02 is recommended value')
    argparser.add_argument('--seed', type=int, default=42)

    args = argparser.parse_args()

    logger = logging.getLogger("httpx")
    logger.setLevel(logging.WARNING)  # Set to WARNING or ERROR to reduce verbosity

    process(args)


if __name__ == '__main__':
    main()
