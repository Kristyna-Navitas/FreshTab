from collections import Counter

import pytest

from dataset_creation import filter_tables, add_labels_to_tables, get_category_distribution, \
    get_distribution_recursively, filter_categories


@pytest.fixture
def sample_dataset():
    return {
        "1": {
            "title": "Page1",
            "category": "sports",
            "table": [
                ["col1", "col2"],
                ["a", "b"],
                ["c", "d"]
            ],
            "table_text": "col1 | col2<br>a | b<br>c | d"
        },
        "2": {
            "title": "Page1",  # Same page, different table
            "category": "sports",
            "table": [
                ["col1", "col2", "col3"],
                ["a", "b", "c"],
                ["d", "e", "f"],
                ["g", "h", "i"]
            ],
            "table_text": "col1 | col2 | col3<br>a | b | c<br>d | e | f<br>g | h | i"
        },
        "3": {
            "title": "Page2",
            "category": "politics",
            "table": [
                ["col1", "col2", "col3", "col4", "col5"],
                ["a", "b", "c", "d", "e"],
                ["e", "d", "c", "b", "a"],
                ["a", "b", "c", "d", "e"]
            ],
            "table_text": "col1 | col2<br>a | b"
        }
    }


@pytest.fixture
def config():
    return {
        "choose_ratio": 0.75,
        "table_max_char": 100,
        "expected_labels": ['superlative', 'aggregation', 'negation', 'comparative', 'unique', 'all']
    }


def test_filter_tables_basic(sample_dataset, config):
    filtered_tables, categories = filter_tables(sample_dataset, config)

    # Should select one table per page
    assert len(filtered_tables) == 2

    # Should have correct category distribution
    assert categories['sports'] == 1
    assert categories['politics'] == 1


def test_filter_tables_prefers_larger_tables(sample_dataset, config):
    config["choose_ratio"] = 1.0  # Always choose larger tables
    filtered_tables, _ = filter_tables(sample_dataset, config)

    # Should select table "2" for Page1 as it's larger
    assert 2 in filtered_tables.keys()


def test_filter_tables_handles_empty_tables():
    empty_dataset = {
        "1": {
            "title": "Page1",
            "category": "sports",
            "table": [],
            "table_text": ""
        }
    }
    filtered_tables, categories = filter_tables(empty_dataset, {"choose_ratio": 1.0, "table_max_char": 100})
    assert len(filtered_tables) == 0


def test_filter_tables_truncates_large_tables(sample_dataset, config):
    config["table_max_char"] = 10  # Very small limit to force truncation
    filtered_tables, _ = filter_tables(sample_dataset, config)

    for table in filtered_tables.values():
        assert len(table['table_text']) <= config["table_max_char"] * 2  # Allow some flexibility


def test_add_labels_to_tables_basic(sample_dataset, config):
    labeled_tables = add_labels_to_tables(sample_dataset, config)

    # Check all tables have labels added
    assert all('labels' in table for table in labeled_tables.values())

    # Check number of labels
    for table in labeled_tables.values():
        assert len(table['labels']) == 5
    # Check labels are unique within each table
    for table in labeled_tables.values():
        assert len(set(table['labels'])) == len(table['labels'])
    # Check all labels are from expected set
    expected_labels_set = set(config['expected_labels'])
    for table in labeled_tables.values():
        assert all(label in expected_labels_set for label in table['labels'])


def test_get_category_distribution_basic():
    categories = {'cat1': 10, 'cat2': 20, 'cat3': 30}
    result = get_category_distribution(15, categories)

    assert sum(result.values()) == 15
    assert all(result[cat] <= categories[cat] for cat in categories)


def test_get_category_distribution_empty():
    assert get_category_distribution(10, {}) == {}


def test_get_category_distribution_insufficient():
    categories = {'cat1': 2, 'cat2': 3}
    result = get_category_distribution(10, categories)

    assert sum(result.values()) == 5
    assert result == {'cat1': 2, 'cat2': 3}


def test_get_distribution_recursively():
    categories = {
        'sport | football': 10,
        'sport | basketball': 5,
        'news | politics': 8,
        'news | weather': 7
    }

    result = get_distribution_recursively(20, categories)

    assert sum(result.values()) == 20
    assert all(cat in categories for cat in result)


def test_filter_categories():
    tables = {
        '0': {'category': 'sport | baseball', 'csv_id': '1.csv', 'table_text': "blablabla"},
        '1': {'category': 'sport | football', 'csv_id': '1.csv', 'table_text': "blablabla"},
        '2': {'category': 'sport | football', 'csv_id': '2.csv', 'table_text': "blablabla"},
        '3': {'category': 'news | politics', 'csv_id': '3.csv', 'table_text': "blablabla"},
        '4': {'category': 'news | weather', 'csv_id': '3.csv', 'table_text': "blablabla"},
        '5': {'category': 'news | politics', 'csv_id': '3.csv', 'table_text': "blablabla"},
    }

    categories = Counter(['sport | baseball', 'sport | football', 'news | politics', 'news | weather'])

    config = {
        'num_pages': 4,
        'excluded_csv_ids': ['2.csv']
    }

    result = filter_categories(config, tables, categories)
    print(result)

    assert len(result) == 4
    assert '2' not in result

