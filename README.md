# FreshTab
Creating new datasets for table-to-text task from Wikipedia pages.

## Features
- Automated Wikipedia table extraction, each month a new version
- Multi-category support (sports, culture, politics) with configurable counts
- Logical labels from LogicNLG and LoTNLG
- Configurable data collection parameters

## Comming Soon
- Do not forget to set up the Airflow correctly!
- more tests of the code
- saving config file for each run
- collecting data in more languages for parallel dataset
- LLM calls with Ollama
- choosing logical labels with LLM
- test calls for the LLMs and its evaluation

## Setup
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Then just don't forget to set up the `config.yaml`, at least your email.
And Airflow if you run the DAG...

## Poster
- in the publications folder
- do not look there if this should be anonymized for you
