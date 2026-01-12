import json
import logging
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup

from utils import read_json, create_directory, save_json, table_from_data_entry, query_wikidata, wiki_mod_date, \
    download_page

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

random.seed(config['random_seed'])
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def query_recursively(cnfg: dict, end_date: date, qid: str, prop: str, cat: str, depth: int = 0) -> (dict, bool):
    """
    Query for instances of category recursively with subcategories.
    config: Configuration dictionary
    end_date: End date for query range
    qid: Wikidata QID
    prop: Property to query
    category: Category name
    depth: Current recursion depth (default: 0)
    Returns: Tuple of (entities dictionary, timeout boolean)
    """
    # current category
    entities, timeout = category_query(cnfg, end_date, qid, cat)
    # max depth - mostly okay with 5, MediaWiki pages in software okay with 10,
    # politics still times out for 10, for 15 it cycles!
    if timeout:
        logger.info(f'Query for category {qid}, {cat} time-outs. Dividing it into subcategories.')
        subcats = divide_to_subcats(cnfg, qid, cat)

        if subcats:
            entities = {}
            for subcat, subcat_label in subcats.items():
                subcat_entities, subcat_timeout = query_recursively(cnfg, end_date, subcat, prop,
                                                                    f'{cat} | {subcat_label}', depth + 1)
                entities.update(subcat_entities)

    if depth >= cnfg['max_depth']:
        logger.info(f'Max depth for {cat}.')

    logger.info(f'Found {len(entities)} entries for category {cat}')
    return entities, timeout


def category_query(cnfg: dict, end_date: date, instance_of: str, subcat: str) -> (dict, bool):
    """
    Query instances for one category with date constraints.
    Each instance has to have start between specified date and now,
    and Wikipedia page from which we can get tables.
    We also get properties, but they are not used yet...

    config: Configuration dictionary
    end_date: End date for query range
    instance_of: QID of the instance type
    category: Category name
    Returns: Tuple of (entities dictionary, timeout boolean)
    """
    # filters to divide between categories
    specific_category = {
        'sport': "?item wdt:P641 [].",
        'culture': "FILTER NOT EXISTS {?item wdt:P641 [].}"
    }
    if any(inst in subcat for inst in cnfg['same_for_categories']):
        divide_cat = specific_category.get(subcat.split(' | ')[0], "")
    else:
        divide_cat = ""

    date_filter = f"""{{
    VALUES ?date_prop {{ {' '.join(f'wdt:{prop}' for prop in cnfg['date_properties'])} }} ?item ?date_prop ?date.
    }}
    UNION
    {{?item wdt:P2348 ?period. ?period wdt:P580 ?date.}}
    FILTER (?date > "{cnfg['date']}"^^xsd:dateTime).
    FILTER (?date < "{end_date}"^^xsd:dateTime)."""

    sparql_query = f"""
    SELECT DISTINCT ?item ?date ?wikipediaUrl
    WHERE
    {{
      ?item wdt:P31/wdt:P279* wd:{instance_of}.
      {divide_cat}
      {date_filter}
      ?wikipediaUrl schema:about ?item; schema:isPartOf <https://{cnfg['lang']}.wikipedia.org/>.
    }}
    """
    result = query_wikidata(sparql_query, cnfg['bot_email'])

    if result:
        found_entities = {item['item']['value'].replace('http://www.wikidata.org/entity/', ''):
                              (item['wikipediaUrl']['value'], subcat, item.get('type', {}).get('value', ''))
                          for item in result["results"]["bindings"]}
        return found_entities, False
    else:
        return {}, True  # will query for subcategories


def divide_to_subcats(cnfg, category: str, label: str) -> dict or None:
    """
    Get subcategories of a given category on Wikidata.
    config: Configuration dictionary
    category: Category QID
    parent_label: Parent category label
    Returns: Dictionary of subcategory QIDs to labels, or None if query fails
    """

    no_longtail = f"HAVING (COUNT(DISTINCT ?subitem) > {cnfg['quick_mode']})" if cnfg['quick_mode'] else ''
    cat_query = f"""
    SELECT ?item ?itemLabel (COUNT(DISTINCT ?subitem) AS ?subitemCount)
    WHERE
    {{
      ?item wdt:P279 wd:{category}.
      ?subitem wdt:P279 ?item.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,mul". }}
    }}
    GROUP BY ?item ?itemLabel
    {no_longtail}
    ORDER BY DESC(?subitemCount)
    """

    # extract and filter categories
    result = query_wikidata(cat_query, cnfg["bot_email"])
    if result:
        # we can get a category we already have and get nasty cycles
        parent_labels = set(label.split(' | '))
        found_qids = {item['item']['value'].replace('http://www.wikidata.org/entity/', ''):
                          item['itemLabel']['value'] for item in result["results"]["bindings"]
                      if item['itemLabel']['value'] not in parent_labels}
        return found_qids

    return None


def query_label(cnfg: dict, qid: str, url: str, category) -> dict:
    """
    Query additional details about an entity including labels, dates, and classifications.

    config: Configuration dictionary
    qid: Wikidata QID
    url: Wikipedia URL
    category: Entity category
    Returns: Dictionary containing entity details
    """
    # we want just english labels for now
    label_query = f"""
    SELECT ?itemLabel
    WHERE {{
        wd:{qid} rdfs:label ?itemLabel.
        FILTER (LANG(?itemLabel) = "{cnfg['lang']}") # Filter by language 
    }}"""

    # not only for culture, lets use it for all categories
    dates_query = f"""SELECT ?prop ?propLabel ?date
    WHERE {{
        VALUES ?prop {{ {' '.join(f'wdt:{prop}' for prop in cnfg['date_properties'])} }}
        wd:{qid} ?prop ?date.
    }}"""

    details = {}
    # getting label
    result = query_wikidata(label_query, cnfg['bot_email'])
    if not result or not result['results']['bindings']:  # entity with no label...
        logger.warning(f"No label found for entity {qid}")
        return {}
    details["label"] = result['results']['bindings'][0]['itemLabel']['value']

    # getting all the dates
    dates_result = query_wikidata(dates_query, cnfg['bot_email'])
    if dates_result:
        from_date = datetime.fromisoformat(cnfg['date'].replace("Z", "+00:00"))
        dates = []
        for item in dates_result['results']['bindings']:
            if item.get('date', {}).get('value'):
                try:
                    dates.append(datetime.fromisoformat(item['date']['value'].replace("Z", "+00:00")))
                except ValueError:
                    pass

        if dates:
            details["all_dates_after"] = all(dt > from_date for dt in dates)
        else:
            details["all_dates_after"] = True  # No dates found, assume valid
        if not details["all_dates_after"]:
            logger.warning(f"Some dates are early for entity {qid}")
            return {}
        details["date"] = wiki_mod_date(config, url)

    if category == "people":  # choosing if people are sport, culture or politics
        people_details = f"""
        SELECT ?occupation ?occupationLabel ?sport ?sportLabel ?class ?classLabel
         WHERE {{
            OPTIONAL {{wd:{qid} wdt:P641 ?sport.}}
            wd:{qid} wdt:P106 ?occupation.
            ?occupation wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{cnfg['lang']}".}}
        }}"""

        details_result = query_wikidata(people_details, cnfg['bot_email'])
        if details_result:
            bd = details_result["results"]["bindings"]
            has_sport = any(b.get("sportLabel") for b in bd)
            occ = [b.get("classLabel", "") for b in bd]
            occupation = {c["value"] for c in occ if c}
            details["sport"] = has_sport
            details["occupation"] = occupation

    return details


def format_table(table: pd.DataFrame) -> (list, str):
    """
    Formats the dataframe table for output json into list of lists format and string format.
    df: Pandas DataFrame to format
    Returns: Tuple of (formatted table as list of lists, formatted table as string)
    """

    table.columns = [str(col).strip().lower().replace(' ', '_') for col in table.columns]
    formatted_table = [list(table.columns)] + table.astype(str).values.tolist()
    table_text = ' | '.join(table.columns) + '<br>'
    table_text += '<br>'.join([' | '.join(row) for row in table.astype(str).values])

    return formatted_table, table_text


def remove_refs(text: str) -> str:
    """
    Remove reference tags like [2] from text.
    text: input text
    """
    return re.sub(r"\[\d+](?:\[\d+])*$", "", str(text)).strip()


def clean_text(text: str) -> str:
    """
    Clean text content from HTML artifacts.
    text: Input text
    Returns: Cleaned text
    """
    if pd.isna(text):
        return text
    if any(marker in str(text) for marker in [".mw-parser-output", "@media", "@supports"]):
        matches = re.search(r'<span[^>]*>([^<]*)</span>', text, re.IGNORECASE)
        if matches:
            return ' '.join(matches.groups())
        # remove style blocks
        text = text.split("@supports\\")[0].strip()
        text = text.split("@media screen")[0].strip()
        text = text.split(".mw-parser-output")[0].strip()
    return text


def retrieve_tables(cnfg, url: str) -> (list, str):
    """Gets tables for one Wikipedia page. Excludes tables that are small or big (defined by parameters).
    Lets just save all the okay pages as we have them anyway and decide which to pick later."""
    # get page
    html_content = download_page(url, cnfg['bot_email'])

    # get tables
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})

    formatted_tables = []
    # choosing table - just don't want small and empty ones? - filtering will be done later
    for tbl in tables:
        try:
            table_df = pd.read_html(StringIO(str(tbl)))[0]

            # Handle multicolumns format like ('prizes',_'zar') -> prizes_zar
            if isinstance(table_df.columns, pd.MultiIndex):
                table_df.columns = table_df.columns.map('_'.join)

            # dropping weird rows
            rows_to_drop = []
            for index, row in table_df.iterrows():
                # Check if all values are the same and long strings - those are the merged cells in tables
                if (all(value == row.iloc[0] for value in row) and
                        (isinstance(row.iloc[0], str) and len(row.iloc[0]) > 100)):
                    rows_to_drop.append(index)
            table_df = table_df.drop(index=rows_to_drop)  # Let's not use long texts like description of episode

            # cleaning up references
            table_df.columns = [remove_refs(col) for col in table_df.columns]
            for col in table_df.columns:
                for index, value in table_df[col].items():
                    if value == 'nan':
                        value = np.nan
                    table_df.at[index, col] = remove_refs(remove_refs(value))

            # cleaning up NaNs
            table_df = table_df.replace(cnfg['nan_values'], pd.NA)

            # remove bad rows and columns - only none values, 0 or all same values; except the first row
            table_df = table_df.dropna(how='all')
            columns_to_drop = []
            for col in table_df.columns:
                if table_df[col].iloc[1:].nunique() == 1:
                    columns_to_drop.append(col)
                    continue
                na_cells = table_df[col].iloc[1:].isna().sum() + (table_df[col].iloc[1:] == '0').all()
                if na_cells > (1 - cnfg['nan_threshold']) * len(table_df):  # too many na in a column
                    columns_to_drop.append(col)
            table_df = table_df.drop(columns=columns_to_drop)

            # cleaning weird text rows from table
            table_df = table_df.map(clean_text)

            # and remove unreasonably long (more than tweet :D) strings that are often some maps and pictures
            table_df = table_df.map(lambda x: None if isinstance(x, str) and len(x) > cnfg['max_text_length'] else x)

            if not table_df.shape[1]:
                continue
            table_size = table_df.shape[0] * table_df.shape[1]
            if table_size < 2:
                continue
            missing = table_df.isna().sum().sum() + (table_df == 0).sum().sum()
            nan_ratio = missing / table_size
            score = table_size * (
                    1 - nan_ratio) if nan_ratio != 0 else table_size  # how many values the table really has
            # truncate long tables
            if len(table_df) > cnfg['max_rows']:
                table_df = table_df[:cnfg['max_rows']]

            # filter out terrible tables
            if (len(table_df) >= cnfg['min_rows'] and cnfg['min_columns'] <= len(table_df.columns) <= cnfg[
                'max_columns']
                    and score >= cnfg['min_entries'] and nan_ratio < 0.25):
                formatted_tables.append((format_table(table_df)))

        except (ValueError, IndexError):
            pass

    return formatted_tables


def process_entities(cnfg: dict, entities: dict, all_quids_found: set, subcategories: dict, cat_name: str, index: int) \
        -> (dict, int):
    """Process entities and extract tables."""
    random.shuffle(list(entities.keys()))  # to not take just the first ones
    extracted_tables = {}
    all_wikipages = []
    for ent, (url, subcat, enty_type) in entities.items():
        if ent in all_quids_found or subcategories[subcat] > cnfg['max_pages']:
            continue
        # retrieve their labels, dates, professions...
        details = query_label(cnfg, ent, url, subcat)
        if not details or not details.get('all_dates_after', True):
            # a new release of already released thing, we do not want that
            continue
        if 'date' not in details:
            continue
        if cnfg['date'] <= details['date'] <= cnfg['until_date']:
            all_wikipages += [[ent, details['date'], subcat]]

        tables = retrieve_tables(cnfg, url=url)
        for formatted, text in tables:
            value = {"title": details['label'], "category": subcat, "type": enty_type, "url": url,
                     "date": details["date"], "csv_id": f'{ent}_{index}.csv', "QID": ent,
                     "table": formatted, "table_text": text}
            try:
                json.dumps(value)  # validating JSON serialization
                extracted_tables[index] = value
                if ent not in all_quids_found:
                    subcategories[subcat] += 1
                    all_quids_found.add(ent)
                index += 1
            except TypeError:
                print('wrong dump')
                pass
        #else:
        #    print(ent, 'no details')
    with open(f'datasets/{config["output_dir"]}/new_pages.csv', "a", encoding="utf-8") as f:
        for page in all_wikipages:
            f.write(f'{",".join(page)}\n')
    logger.info(
        f'Updated to {len(extracted_tables)} tables for category {cat_name} from {len(entities)} wikipages.')

    return extracted_tables, index, subcategories, all_quids_found


def query_no_start_date_recursively(cnfg, limit: int, category: str, start_date: str = None) -> [dict, int]:
    """
    Recursively query entities without start dates.

    config: Configuration dictionary
    limit: Maximum number of entities to retrieve
    category: Category to query ('people' or 'mix | wikilists')
    Returns: Dictionary mapping entity IDs to Wikipedia URLs and middle date for querying
    """
    start = start_date if start_date else cnfg['earliest_date']

    wikilist_query = f"""
    SELECT DISTINCT ?item ?modDate ?wikipediaUrl
    WHERE
    {{
      ?item wdt:P31 wd:Q13406463.
      ?item schema:dateModified ?modDate.
      FILTER (?modDate > "{cnfg['date']}"^^xsd:dateTime).
      ?wikipediaUrl schema:about ?item; schema:isPartOf <https://{cnfg['lang']}.wikipedia.org/>.
    }}
    LIMIT {limit}
    """
    people_query = f"""
    SELECT DISTINCT ?item ?modDate ?wikipediaUrl
    WHERE
    {{
      ?item wdt:P31 wd:Q5.
      ?item wdt:P569 ?birthDate.  # historical figures don't have much tables & need to filter somehow
      FILTER (?birthDate > "{start}"^^xsd:dateTime && ?birthDate < "{cnfg['date']}"^^xsd:dateTime).
      FILTER(NOT EXISTS {{?item wdt:P570 []}}) # Filter out individuals with a death date
      ?item schema:dateModified ?modDate.
      FILTER (?modDate > "{cnfg['date']}"^^xsd:dateTime).
      ?wikipediaUrl schema:about ?item; schema:isPartOf <https://{cnfg['lang']}.wikipedia.org/>.
    }}
    LIMIT {limit}
    """
    if category == 'people':
        person = query_wikidata(people_query, cnfg['bot_email'])
        if person:
            result = person['results']['bindings']
    elif category == 'mix | wikilists':
        wikilist = query_wikidata(wikilist_query, cnfg['bot_email'])
        if wikilist:
            result = wikilist['results']['bindings']
    else:
        logger.warning("Wrong category for no date entities")
        return {}

    if result or limit < 10:
        found_entities = {item['item']['value'].replace('http://www.wikidata.org/entity/', ''):
                              item['wikipediaUrl']['value'] for item in result if result}
        return found_entities
    else:
        query_no_start_date_recursively(cnfg, limit // 10, category)  # the limit is the wikilist / people limit
        if category == 'people':
            start_date = datetime.fromisoformat(cnfg['earliest_date'].replace('Z', '+00:00')).date()
            end_date = datetime.fromisoformat(cnfg['date'].replace('Z', '+00:00')).date()
            time_diff = abs(end_date - start_date)
            if time_diff < timedelta(days=10 * 365.25):
                logger.warning("Timeframe too small")
                return {}
            mid_date = (start_date + time_diff // 2).isoformat() + 'T00:00:00Z'
            query_no_start_date_recursively(cnfg, limit, category, mid_date)


def get_no_start_entities(cnfg, limit: int, index: int, category: str) -> (dict, int):
    entities = query_no_start_date_recursively(cnfg, limit, category)
    if not entities:
        logger.warning(f'No entities found for category {category}')
        return {}, index
    date_threshold = datetime.fromisoformat(cnfg['date'].replace("Z", "+00:00"))

    new_pages = {}  # just like filtering of the entities that really have creation after specified date
    all_wikipages = []
    for entity_id, url in entities.items():
        if len(new_pages) >= cnfg['max_pages']:  # we have enough pages with tables
            break
        found_date = datetime.fromisoformat(wiki_mod_date(cnfg, url).replace("Z", "+00:00"))
        if found_date <= date_threshold:
            continue

        print('found', found_date, url)
        tables = retrieve_tables(cnfg, url=url)
        details = query_label(cnfg, entity_id, url, category)
        if not details or not details['label']:  # no label...
            continue
        if 'all_dates_after' in details and details['all_dates_after'] is False:
            continue  # a new release of already released thing, we do not want that
        if 'date' not in details:
            continue
        if cnfg['date'] <= details['date'] <= cnfg['until_date']:
            all_wikipages += [[entity_id, details['date'], category]]

        cat = category
        if category == "people":
            if details["sport"] and details["occupation"] in cnfg['people_categories']['sport']:
                cat = "sport | people"
            elif details["occupation"] in cnfg['people_categories']['culture']:
                cat = "culture | people"
            else:  # it is mostly politicians and military personnel
                cat = "politics | people"

        for formatted, text in tables:
            data = {"title": str(details["label"]), "category": str(cat), "type": "", "url": str(url),
                    "date": str(details["date"]), "csv_id": f'{entity_id}_{len(new_pages)}.csv',
                    "QID": str(entity_id), "table": formatted, "table_text": str(text)}
            try:  # sometimes it is not properly serializable -> testing it early
                json.dumps(data, ensure_ascii=False)
                new_pages[index] = data
                index += 1
            except TypeError as e:
                print(e)
                pass
    with open(f'datasets/{config["output_dir"]}/new_pages.csv', "a", encoding="utf-8") as f:
        for page in all_wikipages:
            f.write(f'{",".join(page)}\n')
    logger.info(f'Found {len(entities)} entities for category {category}')

    return new_pages, index


def get_new_dataset(cnfg: dict, data: dict) -> (dict, dict):  # TODO make tests for dataset creation
    """
    Creates a new dataset by querying for new entities in each category,
    getting their Wikipedia page and good enough tables from it.

    cnfg: Configuration dictionary
    data: Optional existing dataset to update
    Returns: Tuple of (new dataset dictionary, category counter)
    """

    new_tables, top_cats = data or {}, {}
    index = int(max([str(key) for key in new_tables.keys()])) if new_tables else 0
    all_quids_found = {table['QID'] for table in new_tables.values() if 'QID' in table}
    end_date = cnfg['until_date'] if cnfg['until_date'] else datetime.now().date()

    if cnfg['load_data']:
        top_cats = [table['category'].split(' | ')[0] for table in new_tables.values() if 'category' in table]
        people_cat = any([True for table in new_tables.values()
                          if 'category' in table and 'people' in table['category']])
        wikilists_cat = any([True for table in new_tables.values()
                            if 'category' in table and 'wikilists' in table['category']])
        subcategories = Counter(table['category'] for table in data.values() if 'category' in table)
    else:
        subcategories, people_cat = {}, False

    # can't think up any other reasonable categories...
    # the ordering matters, the item will be in the first category encountered
    categories = cnfg['categories']
    for cat_name, cat_items in categories.items():
        if cnfg['load_data'] and cat_name in top_cats:
            continue
        logger.info(f'Retrieving items for {cat_name}.')
        tables_for_category = {}
        for qid, prop, name in cat_items:
            # gets promising wikipages, categories will be divided to smaller ones if the query time-outs
            entities, timeout = query_recursively(cnfg, end_date, qid, prop, f'{cat_name} | {name}', 0)

            # retrieves, chooses and formats tables
            logger.info('Retrieving the tables.')
            subcategories.update({e[1]: 0 for e in entities.values()})  # update with the newly found entities
            extracted_tables, index, subcategories, all_quids_found = process_entities(
                cnfg, entities, all_quids_found, subcategories, cat_name, index)
            tables_for_category.update(extracted_tables)

        # saves them, each category its json
        save_json(tables_for_category, f'datasets/{cnfg["output_dir"]}/{cat_name}.json')
        new_tables.update(tables_for_category)

    # queries for stuff without date property
    if cnfg['no_people']:
        return new_tables, subcategories

    # ~ 14,6k of wikilists with modDate!  # like probably just 1 table for 100?
    if not cnfg['load_data'] or (cnfg['load_data'] and not wikilists_cat): #'mix' not in top_cats):
        wikilist_tables, index = get_no_start_entities(cnfg, cnfg['wikilist_limit'], index, 'mix | wikilists')
        subcategories['mix | wikilists'] = len(wikilist_tables)
        save_json(wikilist_tables, f'datasets/{cnfg["output_dir"]}/wikilists.json')
        new_tables.update(wikilist_tables)
        logger.info(f'Updated to {len(wikilist_tables)} tables for category mix | wikilists.')

    if not cnfg['load_data'] or (cnfg['load_data'] and not people_cat):
        people_tables, index = get_no_start_entities(cnfg, cnfg['people_limit'], index, 'people')  # Q5
        save_json(people_tables, f'datasets/{cnfg["output_dir"]}/people.json')
        new_tables.update(people_tables)
        logger.info(f'Updated to {len(people_tables)} tables for category people.')

    return new_tables, subcategories


def add_labels_to_tables(tables: dict, cnfg) -> dict:
    """Adds five random logical operations to each table."""
    for key, table in tables.items():
        table["logical_labels"] = random.sample(cnfg['expected_labels'], 5)
    return tables


def filter_tables(dataset: dict, cnfg) -> [dict, Counter]:
    """
    Filter tables to get one representative table per Wikipedia page,
    usually the biggest one with low ratio of nan values but there is some randomness to it.
    dataset: Dictionary of tables with their metadata
    Returns:
        chosen_tables: Dictionary of filtered tables with metadata
        subcategories: Counter of categories for the filtered tables
    """

    page_groups = defaultdict(list)
    # Group tables by Wikipedia page title
    for page_id, page_data in dataset.items():
        page_groups[page_data['title']].append(page_id)
    logger.info(f'Found {len(page_groups)} unique Wikipedia pages.')

    chosen_tables = {}
    for page_name, group in page_groups.items():
        best_table, best_score = None, -float('inf')
        for table_key in group:
            if table_key not in dataset:
                continue
            data = dataset.get(str(table_key))
            if not data:
                continue
            table_df = table_from_data_entry(data['table'])
            if table_df.empty:
                continue
            # Calculate table quality score
            table_size = table_df.shape[0] * table_df.shape[1]
            nan_ratio = table_df.isna().sum().sum() / table_size
            score = table_size * (1 - nan_ratio)  # non-empty cells

            # random sampling just to not have only the biggest tables...
            if best_score == -float('inf'):
                best_table, best_score = table_key, score
            elif score > best_score and random.random() < cnfg['choose_ratio']:
                best_table, best_score = table_key, score

        # there surely will be at least one table
        if best_table:
            table_to_add = dataset[str(best_table)]
            text_field = table_to_add['table_text']

            # Truncate large tables, not to run out of context window
            if len(text_field) > cnfg['table_max_char']:
                # number of rows / ( how many times max char in whole string)
                num_rows = math.ceil(len(table_to_add['table']) / (len(text_field) / cnfg['table_max_char']))
                table_to_add['table_text'] = '<br>'.join(text_field.split('<br>')[:num_rows])
                table_to_add['table'] = table_to_add['table'][:num_rows]

            chosen_tables[best_table] = table_to_add

    # Add labels to the tables
    logger.info(f'Selected {len(chosen_tables)} tables after filtering.')
    subcategories = Counter([table['category'] for table in chosen_tables.values() if 'category' in table])

    return chosen_tables, subcategories


def filter_categories(cnfg, tables: dict, categories: dict) -> dict:
    """
    Filter tables to achieve equal distribution across categories.

    tables: Dictionary of tables with their metadata
    categories: Counter object with category counts
    cnfg: Configuration dictionary containing settings
    Returns: Dictionary of filtered tables
    """
    categories = {k: v for k, v in categories.items() if v > 0}
    # get target distribution
    distributions = get_distribution_recursively(cnfg, cnfg['num_pages'], categories)

    # Select tables based on distribution
    new_tables_id = []
    for ctgr, num in distributions.items():
        if cnfg['excluded_csv_ids']:
            tables_in_subcat = [k for k, v in tables.items() if 'category' in v and v['category'] == ctgr
                                and v['csv_id'] not in cnfg['excluded_csv_ids']]
        else:
            tables_in_subcat = [k for k, v in tables.items() if 'category' in v and v['category'] == ctgr]
        random.shuffle(tables_in_subcat)
        new_tables_id.append(tables_in_subcat[:num])

    new_tables_id = [item for sublist in new_tables_id for item in sublist]
    new_tables = {key: tables[key] for key in new_tables_id if key in tables}

    logger.info(f'Tables filtered to {len(new_tables)} tables.')

    logger.info(
        f'Categories distribution: {Counter(value["category"].split(" | ")[0] for value in new_tables.values())}')

    return new_tables


def get_distribution_recursively(cnfg: dict, target_number: int, categories: dict, level: int = 0) -> dict:
    """
    Recursively distribute target count across category hierarchy.

    cnfg: Configuration dictionary containing settings
    categories: Dictionary of category counts
    level: Current recursive level
    Returns: Dictionary of category distributions
    """
    level_cats = {" | ".join(part.split(' | ')[:level + 1]).strip() for part in categories}
    level_cat_dict = {c: sum(v for k, v in categories.items() if k.startswith(c)) for c in level_cats}
    # get current level distribution
    if cnfg['custom_distribution'] and level == 0:
        # use custom distribution if provided
        real_vals_in_cat = {cat: cnfg['custom_distribution'].get(cat, 0) for cat in level_cat_dict}
    else:
        real_vals_in_cat = get_category_distribution(target_number, level_cat_dict)

    # subcategories recursively
    final_distribution = {}
    for cat, num in real_vals_in_cat.items():
        filtered_cats = {k: v for k, v in categories.items() if k.startswith(cat) and k != cat}
        if not filtered_cats:
            final_distribution.update({cat: num})
        elif (len(filtered_cats) == 1) or (level + 2 == max(len(part.split(" | ")) for part in filtered_cats)):
            final_distribution.update(get_category_distribution(num, filtered_cats))
        else:
            final_distribution.update(get_distribution_recursively(cnfg, num, filtered_cats, level + 1))

    return final_distribution


def get_category_distribution(target_pages: int, categories: dict) -> dict:
    """
    Distributes target count across categories.

    target_count: Number of items to distribute
    categories: Dictionary of category counts
    Returns: Dictionary of category distributions
    """
    if not categories:
        return {}

    # get the ideal count for each category
    ideal_page_count = target_pages // len(categories)
    real_vals_in_cat = {}

    # first pass: assign up to ideal count
    for category, available in categories.items():
        assigned = min(ideal_page_count, available)
        real_vals_in_cat[category] = assigned
        categories[category] = available - assigned

    # second pass: add the reminder from the categories with more entries
    remainder = target_pages - sum(real_vals_in_cat.values())
    if remainder == 0:
        return real_vals_in_cat

    remaining_cats = {k: v for k, v in categories.items() if v > 0}
    for i, cat in enumerate(sorted(remaining_cats.keys(), key=lambda c: remaining_cats[c])):
        if remainder == 0:
            break
        to_add = min(remainder // (len(remaining_cats) - i), categories[cat])
        real_vals_in_cat[cat] += to_add
        remainder -= to_add

    return real_vals_in_cat


def main():
    """Gets the pages and saves tables for them in separate category files,
    then filters tables so each page has only one table and saves all of it together,
    and if we want a specific number of tables, it diversely chooses them across categories and saves those."""

    input_path = Path('datasets') / Path(config["output_dir"])
    # save the current config to the output directory to be sure about what we did
    if not input_path.exists():
        create_directory(str(input_path))
        save_json(config, str(input_path) + '/config.json')

    # TODO more languages together!

    if config['load_data'] and input_path.exists():
        logger.info("Loading existing dataset...")
        new_dataset = {}
        counter = 1
        for file in os.listdir(input_path):
            if file.endswith('.json') and 'config' not in file:
                data = read_json(Path(input_path / file))
                for k, v in data.items():
                    new_key = k
                    while new_key in new_dataset:
                        new_key = f"{k}_{counter}"
                        counter += 1
                    new_dataset[new_key] = v

        logger.info("Processing new tables...")
        new_dataset, subcategories = get_new_dataset(config, new_dataset)
    else:
        logger.info("Processing new tables...")
        new_dataset, subcategories = get_new_dataset(config, {})

    # saves all reasonable tables (we are looking at them anyway) and chooses afterward
    logger.info("Filtering tables and saving dataset...")
    filtered_tables, subcategories = filter_tables(new_dataset, config)
    # add logical operations to the tables
    tables_with_labels = add_labels_to_tables(filtered_tables, config)  # TODO choose them with LLM ?
    save_json(tables_with_labels, 'datasets/' + config['output_dir'] + '/filtered_tables.json')
    if config["num_pages"] or config["custom_distribution"]:
        # we need to know what categories we have for distributing amongst them
        final_tables = filter_categories(config, filtered_tables, subcategories)  # return is just for tests

        save_json(final_tables, 'datasets/' + config['output_dir'] + '/diverse.json')
        # putting the tables in csv to a folder
        create_directory('datasets/' + config['output_dir'] + '/all_csv')
        for table in final_tables.values():
            write_table = table['table_text'].replace("#", '').replace(" | ", "#").replace("<br>", "\n")
            with open(f'datasets/{config["output_dir"]}/all_csv/{table["csv_id"]}', "w", encoding="utf-8") as f:
                f.write(write_table)

    logger.info("Dataset processing complete.")
    # Tsave the config yaml as well just to be sure you do not forget the settings
    shutil.copy(Path('config.yaml'), Path('datasets/' + config['output_dir']))


if __name__ == "__main__":
    main()

