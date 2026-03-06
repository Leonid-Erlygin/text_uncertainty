import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def compute_best_values(table, metric_order, round_num=3):  # Added round_num parameter
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    newdf = table.select_dtypes(include=numerics)
    best_values = {}
    for column_name in newdf.columns:
        if column_name not in metric_order or metric_order[column_name] == "high":
            sorted_values = np.sort(table[column_name].values)[::-1]
        elif metric_order[column_name] == "low":
            sorted_values = np.sort(table[column_name].values)
        else:
            raise ValueError
        # Round the best values before storing
        best_values[column_name] = (np.round(sorted_values[0], round_num), np.round(sorted_values[1], round_num))
    return best_values


def create_table_head(result_latex_code, caption, table_lable, cfg):
    used_columns = cfg.used_columns_dict[cfg.task]
    column_count = 0
    fars = [""]
    num_fars = len(used_columns[cfg.datasets[0]]) - 1
    for key in used_columns:
        if key not in cfg.datasets:
            continue
        for column in used_columns[key]:
            if column == "models":
                continue
            far = column.split("=")[-1]
            fars.append(f"${far}$")
        column_count += len(used_columns[key]) - 1

    if cfg.fix_table:
        result_latex_code += "\\begin{table}[H]\n"
    else:
        result_latex_code += "\\begin{table}\n"
    result_latex_code += "\\caption{" + caption + "}\n"
    result_latex_code += "\\label{" + table_lable + "}\n"
    result_latex_code += f"\\{cfg.table_size}\n"
    result_latex_code += "\\centering\n"
    result_latex_code += "\\setlength\\tabcolsep{" + str(cfg.tabcolsep) + "pt}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\begin{adjustbox}{width=0.5\\textwidth}\n"

    result_latex_code += "\\begin{tabular}{" + "l" + "c" * column_count + "}\n"
    result_latex_code += "\\toprule\n"
    result_latex_code += (
        "Method & \multicolumn{"
        + str(column_count)
        + "}{c}{$\mathrm{FPIR}$}"
        + " \\\\\n"
    )
    result_latex_code += " & ".join(fars) + " \\\\\n"
    result_latex_code += "\\midrule\n"
    if len(cfg.datasets) > 1:
        result_latex_code += " "
        for dataset in cfg.datasets:
            dataset_pretty_name = cfg.pretty_name.dataset[dataset]
            result_latex_code += "& \\multicolumn{"+str(num_fars)+"}{c}{" + dataset_pretty_name + "} "
        result_latex_code += "\\\\\n"
        result_latex_code += "\\midrule\n"
    return result_latex_code


def create_table_body(result_latex_code, cfg):
    dataset_to_metrics = {}
    dataset_to_best_values = {}
    for dataset in cfg.datasets:
        used_columns = OmegaConf.to_container(cfg.used_columns_dict)[cfg.task][dataset]
        table_path = cfg.metric_table_path.format(dataset=dataset)
        all_metric_values = pd.read_csv(table_path)
        all_metric_values = all_metric_values.drop(
            all_metric_values[
                (all_metric_values["models"].str.contains("Random"))
                | (all_metric_values["models"].str.contains("Oracle"))
            ].index
        )
        dataset_to_metrics[dataset] = all_metric_values[used_columns]
        dataset_to_best_values[dataset] = compute_best_values(
            all_metric_values[used_columns], cfg.metric_order, cfg.round_num  # Pass round_num
        )
    table = dataset_to_metrics[cfg.datasets[0]].set_index("models")
    best_values = pd.DataFrame(dataset_to_best_values[cfg.datasets[0]])
    for i in range(len(cfg.datasets) - 1):
        next_table = dataset_to_metrics[cfg.datasets[i + 1]]
        next_table = next_table.set_index("models")
        table = pd.concat([table, next_table], axis=1)
        next_table_best = pd.DataFrame(dataset_to_best_values[cfg.datasets[i + 1]])
        best_values = pd.concat([best_values, next_table_best], join="inner", axis=1)
    table = table.reset_index()
    for row_index, (_, row) in enumerate(table.iterrows()):
        for column_index, column_name in enumerate(table.columns):
            if column_name == "models":
                if "beta" in row[column_name]:
                    result_latex_code += row[column_name].split("-")[-1] + " & "
                else:
                    result_latex_code += cfg.pretty_name.model[row[column_name]] + " & "
            elif column_name != "models":
                # metric value - round first before comparison
                metric_value = row.iloc[column_index]
                rounded_value = np.round(metric_value, cfg.round_num)  # Round first
                if rounded_value == best_values.iloc[0, column_index - 1]:  # Compare rounded values
                    # best value
                    result_latex_code += (
                        "\\textbf{" + str(rounded_value) + "} "
                    )
                elif rounded_value == best_values.iloc[1, column_index - 1]:  # Compare rounded values
                    # second best value
                    result_latex_code += (
                        "\\underline{"
                        + str(rounded_value)
                        + "} "
                    )
                else:
                    result_latex_code += (
                        f" {str(rounded_value)} "
                    )
                if column_index < len(table.columns) - 1:
                    result_latex_code += "& "
                else:
                    # end of row
                    result_latex_code += "\\\\\n"
    return result_latex_code


def create_table_tail(result_latex_code, cfg):
    result_latex_code += "\\bottomrule\n"
    result_latex_code += "\\end{tabular}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\end{adjustbox}\n"
    result_latex_code += "\\end{table}\n"

    return result_latex_code


@hydra.main(
    config_path="/app/configs/latex_tables",
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def run(cfg):
    result_latex_code = """"""

    # table head
    result_latex_code = create_table_head(
        result_latex_code,
        cfg.caption,
        cfg.table_lable,
        cfg,
    )

    # table body
    result_latex_code = create_table_body(result_latex_code, cfg)

    # table tail
    result_latex_code = create_table_tail(result_latex_code, cfg)

    # save result
    hydra_cfg = HydraConfig.get()
    with open(str(Path(cfg.exp_dir) / f"{hydra_cfg.job.config_name}.tex"), "w") as fd:
        fd.write(result_latex_code)
    print("Out file:")
    print(str(Path(cfg.exp_dir) / f"{hydra_cfg.job.config_name}.tex"))


if __name__ == "__main__":
    run()
