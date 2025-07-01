import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Any

import pandas as pd
from logzero import logger
from ollama import Client, ResponseError
from pydantic import ValidationError


def dump_to_logicnlg_output(in_csv, out_json, prettify=True):

    dataframe = pd.read_csv(in_csv, header=None, na_values=['', ' ', 'nan'])
    d = defaultdict(set)
    # this should now work for both with and without header files
    if dataframe.iloc[0, 0] == "csv_ids":
        dataframe.columns = dataframe.iloc[0]
    else:
        dataframe.columns = ["csv_ids", "preprocess", "label", "idea", "query", "result", "prediction"]
    for _, row in dataframe.iterrows():
        prediction = row["prediction"]
        if isinstance(prediction, str):
            d[row["csv_ids"]].add(prediction)
        else:
            print(f"WARNING: prediction is {type(prediction)=} {prediction}")
    d = {k: list(v) for k, v in d.items()}
    with open(out_json, 'w') as file:
        if prettify:
            json.dump(d, file, indent=4)
        else:
            json.dump(d, file)
    return out_json


@contextmanager
def csv_writer(out_file: str):
    csvfile = open(out_file, "a", newline="")
    writer = csv.writer(csvfile)
    try:
        yield writer
    finally:
        csvfile.close()


class OllamaClient:
    def __init__(self, host: str, model, decoding_options: Dict[str, Any]):
        self.model = model
        self._ollama_client = Client(host=host)
        self._decoding_options = dict(decoding_options)

    def __call__(self, system_msg, prompt, format=None, **decoding_options,):
        # TODO tfs_z https://www.trentonbricken.com/Tail-Free-Sampling/
        assert dict(decoding_options) == {}, f"Only decoding options specified in constructor are supported"
        decoding_options = {**self._decoding_options, **decoding_options}
        prompt = f"{system_msg}\n\nTask: {prompt}"
        try:
            response = self._ollama_client.generate(model=self.model, prompt=prompt, format=format, options=decoding_options)
        except ResponseError as e:
            response = ''
        return response['response'], prompt

    def prompt_probability(self, prompt, seed=4242):
        # TODO return nbest list
        raise NotImplementedError("Track https://github.com/ollama/ollama/issues/2415")
        response = self._ollama_client.generate(model=self.model, prompt=prompt, options={
            'temperature': 0.0,
            "num_predict": 0,
            "logprobs": True,
            'seed': seed,
        })
        __import__('ipdb').set_trace()
        log_probs = response["log_probs"]
        return log_probs


class OllamaClientStructuredOutput(OllamaClient):
    def __init__(self, pydantic_answer_class, host: str, model, decoding_options: Dict[str, Any]):
        super().__init__(host, model, decoding_options)
        self.pydantic_answer_class = pydantic_answer_class
        logger.debug(f"{self.schema=}")

    @property
    def schema(self):
        return self.pydantic_answer_class.model_json_schema()

    def parse_raw(self, raw_reply):
        error_str, structured_output = "", None
        try:
            structured_output = self.pydantic_answer_class.parse_raw(raw_reply.strip())
        except ValidationError as e:
            error_str = f"{e}"
            logger.error(f"{error_str=}")
        return structured_output, error_str

    def __call__(self, system_msg, prompt, format=None):
        schema = self.schema if format is None else format
        return super().__call__(system_msg, prompt, format=schema)

    def run_and_parse(self):
        raw_reply, prompt_str = self()
        logger.debug(f"{raw_reply=}")
        structured_output, error_str = self.parse_raw(raw_reply)
        return raw_reply, structured_output, error_str, prompt_str


def dump_to_LogicNLG_output(in_csv, out_json, prettify=True):

    dataframe = pd.read_csv(in_csv, header=None, na_values=['', ' ', 'nan'])
    d = defaultdict(set)
    # this should now work for both with and without header files
    if dataframe.iloc[0, 0] == "csv_ids":
        dataframe.columns = dataframe.iloc[0]
    else:
        dataframe.columns = ["csv_ids", "preprocess", "label", "idea", "query", "result", "prediction"]
    for _, row in dataframe.iterrows():
        prediction = row["prediction"]
        if isinstance(prediction, str):
            d[row["csv_ids"]].add(prediction)
        else:
            print(f"WARNING: prediction is {type(prediction)=} {prediction}")
    d = {k: list(v) for k, v in d.items()}
    with open(out_json, 'w') as file:
        if prettify:
            json.dump(d, file, indent=4)
        else:
            json.dump(d, file)


def dataframe_table_from_dataset(data: dict) -> pd.DataFrame:
    """ Creates dataframe from the textual representation in dataset
    and cleans the column values, so they are easier to work with in SQL. """

    columns = data[0]
    columns = [c.replace(' ', '_') for c in columns]
    columns = [c.replace('/', '_') for c in columns]
    columns = [c.replace('(', '_') for c in columns]
    columns = [c.replace(')', '_') for c in columns]
    columns = [c.replace('__', '_') for c in columns]
    columns = ['_' + c if c and c[0].isdigit() else c for c in columns]

    rows = data[1:]

    df = pd.DataFrame(rows, columns=columns)

    for column in df.columns:
        try:
            df[column] = df[column].astype(int)
        except Exception as e:
            logger.debug(f"Column {column} could not be converted to int: {e}")
            # If it fails, we keep it as string
            df[column] = df[column].astype(str)

    return df


# inspired by example from LLM-T2T
# See https://github.com/yale-nlp/LLM-T2T/blob/main/prompts/open_src_model/LoTNLG/prompt_LoTNLG_direct_CoT.txt
def format_dataframe(df: pd.DataFrame) -> str:
    # Convert the DataFrame to a string with a custom separator
    header = " | ".join(df.columns)
    rows = df.apply(lambda row: " | ".join(row.astype(str)), axis=1)
    formatted_table = f"{header}\n" + "\n".join(rows)
    return formatted_table
