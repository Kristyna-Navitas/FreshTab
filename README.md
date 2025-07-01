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
- collecting data in more languages for parallel dataset
- choosing logical labels with LLM
- evaluation

## Setup
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to start the data collection, before adjust the `config.yaml` file, at least the email.
```bash
python dataset_creation.py
```
Then there is also generation script for ollama LLM outputs for the dataset in the generation directory.
And evaluation script in the evaluation directory.

## Poster
- in the publications folder
- do not look there if this should be anonymized for you
