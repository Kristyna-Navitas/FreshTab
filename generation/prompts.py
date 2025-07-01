SYSTEM_MSG = "You are an excellent senior editor of a data-driven newspaper."

# LoTNLG prompt_LoTNLG_table2text_zero_shot.txt - for GPT
PROMPT_CHOICE = """\
    Your task is to provide 5 different consistent claims derived from a table according to the given corresponding logical labels.
    Consistent means that all information of your claims should be supported by the corresponding table.
    
    Table Title: '{title}'
    Table columns: {table_columns}
    Input Table:
    {table}
    
    There are 9 kinds of logical labels, the detailed explanation of these logical type labels is as follows:
    - 'aggregation': the aggregation operation refers to sentences like "the averaged age of all ....", "the total amount of scores obtained in ...", etc.
    - 'negation': the negation operation refers to sentences like "xxx did not get the best score", "xxx has never obtained a score higher than 5".
    - 'superlative': the superlative operation refers to sentences like "xxx achieves the highest score in", "xxx is the lowest player in the team".
    - 'count': the count operation refers to sentences like "there are 3 xxx from xxx country in the game".
    - 'comparative': the comparative operation refers to sentences like "xxx has a higher score than yyy".
    - 'ordinal': the ordinal operation refers to sentences like "the first country to achieve xxx is xxx", "xxx is the second oldest person in the country".
    - 'unique': the unique operation refers to sentences like "there are 5 different nations in the tournament, ", "there are no two different players from U.S"
    - 'all': the for all operation refers to sentences like "all of the trains are departing in the morning", "none of the people are older than 25."
    - 'none': the sentences which do not involve higher-order operations like "xxx achieves 2 points in xxx game", "xxx player is from xxx country".
    
    Produce specific, informative, and factual insights based on the table above, grounded in valid data points or trends.
    Output {num_ideas} insights. For each insight output also the insight type. Each insight should be a single sentence.
    Provided insights should be different from each other.
    
    Use the json schema to format the output:
    <|json_schema|>
    {ideas_schema}
    <|end_schema|>
    Make sure you only provide insights related to the Input Table.
    
    Think step-by-step about general patterns, then formulate an insight together with its label.
    Output {num_ideas} insights and labels pairs:
    """

# LoTNLG prompt_LoTNLG_direct_CoT.txt
PROMPT_COT = """\
Example 1:
Title: 1941 vfl season
Table columns: home team | home team score | away team | away team score | venue | crowd | date
Input Table:
richmond | 10.13 (73) | st kilda | 6.11 (47) | punt road oval | 6000 | 21 june 1941
hawthorn | 6.8 (44) | melbourne | 12.12 (84) | glenferrie oval | 2000 | 21 june 1941
collingwood | 8.12 (60) | essendon | 7.10 (52) | victoria park | 6000 | 21 june 1941
carlton | 10.17 (77) | fitzroy | 12.13 (85) | princes park | 4000 | 21 june 1941
south melbourne | 8.16 (64) | north melbourne | 6.6 (42) | lake oval | 5000 | 21 june 1941
geelong | 10.18 (78) | footscray | 13.15 (93) | kardinia park | 5000 | 21 june 1941

Task: Provide a consistent claim sentence from the table above according to the logical label.
Logical label: superlative
Reasoning: looking at both "home team score" column and "away team score" column, finding the highest score was 13.15 (93) in "away team score" column and then looking for which team scored 13.15 (93) in "away team" colmun, footscray scored the most point of any team that played on 21 june.
Claim: footscray scored the most point of any team that played on 21 june, 1941.

#

Example 2:
Title: 2008 universitario de deportes season
Table columns: nat | name | moving to | type | transfer window
Input Table:
per | rivas | górnik zabrze | transfer | winter
per | v zapata | alianza atlético | transfer | winter
per | tragodara | atlético minero | loaned out | winter
per | correa | melgar | loaned out | winter
per | curiel | alianza atlético | transfer | winter

Task: Provide a consistent claim sentence from the table above according to the logical label.
Logical label: all
Reasoning: looking at "transfer window" column, all of the transfer windows were winter.
Claim: all of the transfer window for the 2008 universitario de deportes season were winter.

#

Example 3:
Table Title: '{title}'
Table columns: {table_columns}
{table}

Task: Provide a consistent claim sentence from the table above according to the logical label.
Use the json schema to format the output:
<|json_schema|>
{ideas_schema}
<|end_schema|>
"""

PROMPT_WITH_LABEL = PROMPT_COT + """\
Logical label: {logical_label}: {logical_label_explanation}
"""

explanation = {
    'aggregation': 'the aggregation operation refers to sentences like "the averaged age of all ....", "the total amount of scores obtained in ...", etc.',
    'negation': 'the negation operation refers to sentences like "xxx did not get the best score", "xxx has never obtained a score higher than 5".',
    'superlative': 'the superlative operation refers to sentences like "xxx achieves the highest score in", "xxx is the lowest player in the team".',
    'count': 'the count operation refers to sentences like "there are 3 xxx from xxx country in the game".',
    'comparative': 'the comparative operation refers to sentences like "xxx has a higher score than yyy".',
    'ordinal': 'the ordinal operation refers to sentences like "the first country to achieve xxx is xxx", "xxx is the second oldest person in the country".',
    'unique': 'the unique operation refers to sentences like "there are 5 different nations in the tournament, ", "there are no two different players from U.S"',
    'all': 'the for all operation refers to sentences like "all of the trains are departing in the morning", "none of the people are older than 25."',
    'none': 'the sentences which do not involve higher-order operations like "xxx achieves 2 points in xxx game", "xxx player is from xxx country".'
}
