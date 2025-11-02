import json
import logging
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def create_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory_path)
    except FileExistsError:
        logger.info(f"Directory '{directory_path}' already exists.")


def read_json(data_path: Path) -> dict:
    """ Read the dataset to dictionary """
    with open(data_path, "r", encoding="utf-8") as load_file:
        return json.load(load_file)


def save_json(data: dict, file_path: str, indent: int = 4) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def table_from_data_entry(data: dict) -> pd.DataFrame:
    if len(data) < 2:
        return pd.DataFrame()
    dataframe = pd.DataFrame(data[1:], columns=data[0])
    return dataframe.replace('nan', np.nan)


def get_delay(date: datetime) -> int:
    """Gets the time for which we were told not to query Wikidata."""
    if isinstance(date, str):
        return 1
    try:
        timeout = int((date - datetime.now()).total_seconds())
    except ValueError:
        try:
            timeout = int(str(date))
        except TypeError:
            print('Date "', date, '" is a string.')
            timeout = 1  # because whatever
    return timeout


def query_wikidata(query: str, email: str) -> dict or None:
    """Exectures query to Wikidata and gets result."""

    wikidata_endpoint = "https://query.wikidata.org/sparql"
    headers = {'User-Agent': f'Tables_bot/0.0 ({email})'}
    params = {"format": "json", "query": query}
    try:
        response = requests.get(wikidata_endpoint, params=params, headers=headers)
        if response.status_code == 200:
            try:
                return response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError):
                return None
        elif response.status_code == 500:
            return None
        elif response.status_code == 403:
            return None
        elif response.status_code == 429:  # beware not to get banned!
            timeout = get_delay(response.headers['retry-after'])
            logger.info('Timeout {} m {} s'.format(timeout // 60, timeout % 60))
            time.sleep(timeout)
            query_wikidata(query, email)
    except requests.exceptions.ChunkedEncodingError:
        return None


def wiki_mod_date(cnfg, url: str) -> str:
    # gets the first modification of an entity (creation)
    title = urllib.parse.unquote(url.split("wikipedia.org/wiki/")[-1])
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "timestamp",
        "rvdir": "newer",
        "format": "json"
    }
    headers = {'User-Agent': f'Tables_bot/0.0 ({cnfg["bot_email"]})'}
    response = requests.get(f"https://{cnfg['lang']}.wikipedia.org/w/api.php", params=params, headers=headers)
    try:
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        if "revisions" in page:
            return page['revisions'][0]['timestamp']
    except (requests.exceptions.RequestException, json.JSONDecodeError, StopIteration, KeyError):
        return ''

    return ''


def download_page(url: str, email: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/127.0.0.0 Safari/537.36"
    }  # TODO hack, need to repair it
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("Page retrieved.")
        return response.content
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}. Page {url}")
        return ''

