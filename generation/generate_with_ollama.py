import argparse
import json
import logging
from pathlib import Path
from typing import List

from logzero import logger
from pydantic import BaseModel, Field, ValidationError

from generation.helpers import csv_writer, dump_to_LogicNLG_output, OllamaClientStructuredOutput, \
    dataframe_table_from_dataset, format_dataframe
from generation.prompts import SYSTEM_MSG, PROMPT_CHOICE, PROMPT_WITH_LABEL, explanation
from utils import read_json

EXPECTED_LABELS = ['superlative', 'aggregation', 'negation', 'comparative', 'unique', 'all', 'count', 'ordinal', 'none']


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
        logger.info(f"{example['csv_id']=} {example['title']=}")

        final_claims, labels, idea_error, _idea_prompt = generate_structured_claims(
            self.args, self.llm, example['title'], datatable, logical_label=example.get('logical_labels', 'none'))
        logger.debug(f"{len(final_claims)=} {idea_error=}")

        claims = []
        for label, claim in zip(labels, final_claims):
            # generate sentence from the goal and retrieved data; not whole table text, just header
            claims.append([label, "DUMMY", "DUMMY", "DUMMY", claim])
        for i in range(5 - len(claims)):  # to have 5 claims in the output always
            DUMMY_LABEL = "surface"  # existing / valid label which we will use for dummy section
            claims.append([DUMMY_LABEL] + (["DUMMY"] * 4))
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
            logical_label=logical_label,
            logical_label_explanation=explanation[logical_label],
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
    label: str = Field(description=f"Category of the insight. Must be exactly one of: {EXPECTED_LABELS}.")
    insight: str = Field(description="Your data insight idea.")


class InsightsLabelsCoT(BaseModel):
    thoughts: str = Field(description="Your step-by-step thoughts which data insights ideas are worth generating.")
    insight_label_pairs: List[InsightAndLabel]

    @property
    def labels(self):
        return [t.label for t in self.insight_label_pairs]

    @property
    def insights(self):
        return [t.insight for t in self.insight_label_pairs]


def process(args):
    generate_table_claims = Experiment(args)

    dataset = read_json(args.input)
    output_claims_path = Path(args.output)
    logger.info(f"{output_claims_path=}")

    # generate claims
    with csv_writer(output_claims_path) as out_writer:
        for idx, (csv_id, example) in enumerate(dataset.items()):
            logger.debug(f'Generating claims for {idx}-th table {csv_id}')
            claims = generate_table_claims(example)
            for claim in claims:
                print([example['csv_id'], None] + claim)
                out_writer.writerow([example['csv_id'], None] + claim)

            logger.info(f'Generated claims for {idx+1} table {csv_id}. {output_claims_path=}')

    generate_table_claims.final()
    dump_to_LogicNLG_output(output_claims_path, str(output_claims_path) + ".json")

    # save pretty printed arguments to json file
    with open(Path(args.output) / f"{args.output}.args.json", "wt") as f:
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
