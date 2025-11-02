# FreshTab
Creating new datasets for table-to-text task from Wikipedia pages.

[INLG 2025 Proceedings](https://ufal.mff.cuni.cz/~odusek/inlg2025/inlg2025-main/pdf/2025.inlg-main.7.pdf)


## Features
- Automated Wikipedia table extraction, each month a new version
- Multi-category support (sports, culture, politics) with configurable counts
- Logical labels from LogicNLG and LoTNLG
- Configurable data collection parameters
- Generation of insights with Ollama and evaluation with TAPEX and TAPAS

## Comming Soon
- Do not forget to set up the Airflow correctly!
- more tests of the code
- collecting data in more languages for parallel dataset
- choosing logical labels with LLM

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

## Acknowledgements
Funded by the European Union (ERC, NG-NLG, 101039303),
Charles University projects SVV 260 698, and
National Recovery Plan funded project MPO 60273/24/21300/21000 CEDMO 2.0 NPO.
Using resources provided by the LINDAT/CLARIAH-CZ infrastructure (Czech MEYS No. project LM2018101).

