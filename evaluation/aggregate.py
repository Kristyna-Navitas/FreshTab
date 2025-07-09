import os
import json
import pandas as pd

def generate_comparison_tables(json_folder_path: str) -> pd.DataFrame:
    """
    Reads experiment data from JSON files in a specified folder,
    extracts relevant metrics, and compiles them into a pandas DataFrame.
    Args: json_folder_path: The path to the folder containing the JSON experiment files.
    Returns: A pandas DataFrame containing the extracted and organized experiment data.
    """
    all_experiment_data = []

    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(json_folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "statistics" not in data or 'domains' not in data:
                    continue
                if 'domains' not in data:
                    print('domains are missing')
                    continue

                # Initialize a flat dictionary for this experiment's record
                experiment_record = {"Name": data.get("File name", os.path.basename(filename)).replace('generation/outputs/', '').replace('_evaluated.csv', '')}

                # --- Basic Stats & General Metrics (TAPAS, TAPEX) ---
                experiment_record["All Preds"] = data.get("all predictions")
                experiment_record["Empty Preds"] = data.get("empty predictions")
                experiment_record["BLEU4"] = round(data.get("statistics", {}).get("BLEU", {}).get("BLEU4"), 3)
                experiment_record["Avg Length"] = round(data.get("statistics", {}).get("avg_length"), 0)
                experiment_record["Unique Tokens"] = round(data.get("statistics", {}).get("unique_tokens"), 3)
                experiment_record["Entropy"] = round(data.get("statistics", {}).get("entropy"), 3)
                experiment_record["MSTTR"] = round(data.get("statistics", {}).get("msttr"), 3)

                # General TAPAS metrics
                tapas_metrics = data.get("metrics", {}).get("TAPAS", {})
                experiment_record["TAPAS"] = round(tapas_metrics.get("for_whole"),3)
                experiment_record["TAPAS_Upper"] = round(tapas_metrics.get("upper_bound"),3)
                experiment_record["TAPAS_Lower"] = round(tapas_metrics.get("lower_bound"),3)
                experiment_record["TAPAS_NotEmpty"] = round(tapas_metrics.get("not_empty"),3)

                # General TAPEX metrics
                tapex_metrics = data.get("metrics", {}).get("TAPEX", {})
                experiment_record["TAPEX"] = round(tapex_metrics.get("for_whole"),3)
                experiment_record["TAPEX_Upper"] = round(tapex_metrics.get("upper_bound"),3)
                experiment_record["TAPEX_Lower"] = round(tapex_metrics.get("lower_bound"),3)
                experiment_record["TAPEX_NotEmpty"] = round(tapex_metrics.get("not_empty"),3)

                # --- Operations Accuracy ---
                ops_accuracy = data.get("operations_accuracy", {})
                for op_name, value in ops_accuracy.items():
                    # Flatten 'superlative count' to 'superlative_count'
                    col_name = op_name.replace(" ", "_")
                    experiment_record[f"Acc_{col_name}"] = value

                # --- Operations Metrics ---
                ops_metrics = data.get("operations_metrics", {})
                # Dynamically get operation types from the keys (e.g., 'superlative count', 'comparative count')
                # Extract unique operation names (superlative, comparative, etc.)
                operation_types = set()
                for key in ops_metrics.keys():
                    if ' count' in key:
                        operation_types.add(key.replace(' count', ''))

                for op_type in sorted(list(operation_types)): # Sort for consistent column order
                    op_prefix = op_type.replace(" ", "_") # e.g., 'superlative' -> 'superlative' or 'simple' -> 'simple'

                    # Basic counts and percentages for this operation
                    experiment_record[f"Metrics_{op_prefix}_count"] = ops_metrics.get(f"{op_type} count")
                    #experiment_record[f"Metrics_{op_prefix}_not_empty_count"] = ops_metrics.get(f"{op_type} not empty count")
                    #experiment_record[f"Metrics_{op_prefix}_not_empty_percentage"] = ops_metrics.get(f"{op_type} not empty percentage")

                    # Nested metrics (TAPAS, TAPEX) for this operation
                    op_specific_metrics = ops_metrics.get(f"{op_type} metrics", {})

                    tapas_op = op_specific_metrics.get("TAPAS", {})
                    experiment_record[f"Metrics_{op_prefix}_TAPAS"] = round(tapas_op.get("whole"), 3) if tapas_op.get("whole") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPAS_not_empty"] = round(tapas_op.get("not_empty"), 3) if tapas_op.get("not_empty") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPAS_lower"] = round(tapas_op.get("lower_bound"), 3) if tapas_op.get("lower_bound") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPAS_upper"] = round(tapas_op.get("upper_bound"), 3) if tapas_op.get("upper_bound") is not None else None

                    tapex_op = op_specific_metrics.get("TAPEX", {})
                    experiment_record[f"Metrics_{op_prefix}_TAPEX"] = round(tapex_op.get("whole"), 3) if tapas_op.get("whole") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPEX_not_empty"] = round(tapex_op.get("not_empty"), 3) if tapas_op.get("not_empty") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPEX_lower"] = round(tapex_op.get("lower_bound"), 3) if tapas_op.get("lower_bound") is not None else None
                    experiment_record[f"Metrics_{op_prefix}_TAPEX_upper"] = round(tapex_op.get("upper_bound"), 3) if tapas_op.get("upper_bound") is not None else None

                # --- Domain Metrics ---
                domains_metrics = data.get("domains", {})
                for domain_name in ['sport', 'culture', 'politics', 'mix']:
                     experiment_record[f"Metrics_domain_{domain_name}_count"] = domains_metrics.get(f"{domain_name}  count")
                     domain_not_empty_percentage = domains_metrics.get(f"{domain_name}  not empty percentage")
                     experiment_record[f"Metrics_domain_{domain_name}_not_empty_percentage"] = round(domain_not_empty_percentage, 3) if domain_not_empty_percentage is not None else None

                     # Nested metrics (TAPAS, TAPEX) for this domain
                     domain_specific_metrics = domains_metrics.get(f"{domain_name}  metrics", {})
                     tapas_domain = domain_specific_metrics.get("TAPAS", {})
                     experiment_record[f"Metrics_domain_{domain_name}_TAPAS"] = round(tapas_domain.get("whole"), 3) if tapas_domain.get("whole") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPAS_not_empty"] = round(tapas_domain.get("not_empty"), 3) if tapas_domain.get("not_empty") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPAS_lower"] = round(tapas_domain.get("lower_bound"), 3) if tapas_domain.get("lower_bound") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPAS_upper"] = round(tapas_domain.get("upper_bound"), 3) if tapas_domain.get("upper_bound") is not None else None

                     tapex_domain = domain_specific_metrics.get("TAPEX", {})
                     experiment_record[f"Metrics_domain_{domain_name}_TAPEX"] = round(tapex_domain.get("whole"), 3) if tapex_domain.get("whole") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPEX_not_empty"] = round(tapex_domain.get("not_empty"), 3) if tapex_domain.get("not_empty") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPEX_lower"] = round(tapex_domain.get("lower_bound"), 3) if tapex_domain.get("lower_bound") is not None else None
                     experiment_record[f"Metrics_domain_{domain_name}_TAPEX_upper"] = round(tapex_domain.get("upper_bound"), 3) if tapex_domain.get("upper_bound") is not None else None

                all_experiment_data.append(experiment_record)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
            except Exception as e:
                print(f"Warning: An error occurred processing {filename}: {e}. Skipping.")

    full_df = pd.DataFrame(all_experiment_data)

    # --- Define columns for each desired table ---

    # Table 1: General Statistics & Overall TAPAS/TAPEX
    general_stats = [
        "Name", "All Preds", "Empty Preds", "BLEU4",
        "Avg Length", "Unique Tokens", "Entropy", "MSTTR",
        "TAPAS", "TAPAS_Upper", "TAPAS_Lower", "TAPAS_NotEmpty",
        "TAPEX", "TAPEX_Upper", "TAPEX_Lower", "TAPEX_NotEmpty",
    ]
    output_csv_path = f"evaluation/general_stats.csv"
    general = full_df[general_stats]
    general.sort_values(by="Name", inplace=True)
    general.to_csv(output_csv_path, index=False)
    print(f"Table saved to '{output_csv_path}'")

    # Table 2: Operations stats
    basic_count = make_basic_table(full_df, 'Basic counts', '_count', 'Metrics_', '_not_empty')
    numeric_cols = [col for col in basic_count.columns if col != 'Basic counts']
    basic_count['Total'] = basic_count[numeric_cols].sum(axis=1)

    # percentages of accurately followed operations
    percentages_df = full_df.filter(like='Acc_').filter(like='_percentage').copy().round(3)
    percentages_df.columns = [f"{col.replace('Acc_', '').replace('_percentage', '')} %" for col in percentages_df.columns]
    percentages_df = percentages_df.sort_index(axis=1)
    percentages_df.insert(0, 'Accuracy percentage', full_df['Name'])
    if 'everything %' in percentages_df:
        moved_column_data = percentages_df.pop('everything %')
        percentages_df['Total %'] = moved_column_data
    else:
        numeric_cols = [col for col in percentages_df.columns if col != 'Accuracy percentage']
        percentages_df['Total'] = percentages_df[numeric_cols].sum(axis=1)
    percentages_df.sort_values(by='Accuracy percentage', inplace=True)

    # --- TAPAS (Whole) ---
    df_tapas = make_basic_table(full_df, 'TAPAS', '_TAPEX', 'Metrics_', '_not_empty')
    # --- TAPAS Lower Bound ---
    df_tapas_lb = make_basic_table(full_df, 'TAPAS lower bound', '_TAPAS_lower', 'Metrics_', '_not_empty')
    # --- TAPAS Upper Bound ---
    df_tapas_ub = make_basic_table(full_df, 'TAPAS upper bound', '_TAPAS_upper', 'Metrics_', '_not_empty')
    # --- TAPAS Not Empty ---
    df_tapas_ne = make_basic_table(full_df, 'TAPAS not empty', '_TAPAS_not_empty', 'Metrics_', '_not_empty_')
    # --- TAPEX (Whole) ---
    df_tapex = make_basic_table(full_df, 'TAPEX', '_TAPEX', 'Metrics_', '_not_empty')
    # --- TAPEX Lower Bound ---
    df_tapex_lb = make_basic_table(full_df, 'TAPEX lower bound', '_TAPEX_lower', 'Metrics_', '_not_empty')
    # --- TAPEX Upper Bound ---
    df_tapex_ub = make_basic_table(full_df, 'TAPEX upper bound', '_TAPEX_upper', 'Metrics_', '_not_empty')
    # --- TAPEX Not Empty ---
    df_tapex_ne = make_basic_table(full_df, 'TAPEX upper bound', '_TAPEX_not_empty', 'Metrics_', '_not_empty_')

    # Write to single CSV
    output_csv_path = "evaluation/operations.csv"
    with open(output_csv_path, 'w', newline='') as f:
        basic_count.to_csv(f, index=False, header=True)
        f.write("\n")
        #f.write("Accuracy Percentages,-,-,-,-,-,-,-,-,-,-,\n")
        percentages_df.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_ub.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_lb.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_ne.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_ub.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_lb.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_ne.to_csv(f, index=False, header=True)
        f.write("\n")
        print(f"'{output_csv_path}' generated.")

    # Table 3: domains stats
    basic_count = make_basic_table(full_df, 'Basic counts', '_count', 'Metrics_domain_', '_not_empty')
    numeric_cols = [col for col in basic_count.columns if col != 'Basic counts']
    basic_count['Total'] = basic_count[numeric_cols].sum(axis=1)

    df_ne = make_basic_table(full_df, 'Not empty %', '_not_empty_percentage', 'Metrics_domain_', 'whatever')

    # --- TAPAS (Whole) ---
    df_tapas = make_basic_table(full_df, 'TAPAS', '_TAPAS', 'Metrics_domain_', '_not_empty')
    # --- TAPAS Lower Bound ---
    df_tapas_lb = make_basic_table(full_df, 'TAPAS lower bound', '_TAPAS_lower', 'Metrics_domain_', '_not_empty')
    # --- TAPAS Upper Bound ---
    df_tapas_ub = make_basic_table(full_df, 'TAPAS upper bound', '_TAPEX_upper', 'Metrics_domain_', '_not_empty')
    # --- TAPAS Not Empty ---
    df_tapas_ne = make_basic_table(full_df, 'TAPAS not empty', '_TAPAS_not_empty', 'Metrics_domain_', '_not_empty_')
    # --- TAPEX (Whole) ---
    df_tapex = make_basic_table(full_df, 'TAPEX', '_TAPEX', 'Metrics_domain_', '_not_empty')
    # --- TAPEX Lower Bound ---
    df_tapex_lb = make_basic_table(full_df, 'TAPEX lower bound', '_TAPEX_lower', 'Metrics_domain_', '_not_empty')
    # --- TAPEX Upper Bound ---
    df_tapex_ub = make_basic_table(full_df, 'TAPEX upper bound', '_TAPEX_upper', 'Metrics_domain_', '_not_empty')
    # --- TAPEX Not Empty ---
    df_tapex_ne = make_basic_table(full_df, 'TAPEX not empty', '_TAPEX_not_empty', 'Metrics_domain_', '_not_empty_')

    # Write to single CSV
    output_csv_path = "evaluation/domains.csv"
    with open(output_csv_path, 'w', newline='') as f:
        basic_count.to_csv(f, index=False, header=True)
        f.write("\n")
        df_ne.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_ub.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_lb.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapas_ne.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_ub.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_lb.to_csv(f, index=False, header=True)
        f.write("\n")
        df_tapex_ne.to_csv(f, index=False, header=True)
        f.write("\n")
        print(f"'{output_csv_path}' generated.")


    return


def make_basic_table(df: pd.DataFrame, name: str, replace: str, start: str, exclude: str) -> pd.DataFrame:
    df_new_cols = ['Name'] + [col for col in df.columns if col.startswith(start) and col.endswith(replace) and exclude not in col]
    df_new = df[df_new_cols].copy()
    df_new.columns = [name] + [f"{col.replace('Metrics_', '').replace(replace, '')}" for col in df_new.columns if col != 'Name']
    numeric_cols = [col for col in df_new.columns if col != name]
    df_new['Total'] = None
    df_new.sort_values(by=name, inplace=True)
    return df_new


if __name__ == "__main__":
    # --- Configuration ---
    # Replace 'path/to/your/json/folder' with the actual path to your JSON files
    json_folder_path = "generation/outputs/" # Example: "experiment_results" or "data/jsons"

    # Generate the comparison tables
    generate_comparison_tables(json_folder_path)

