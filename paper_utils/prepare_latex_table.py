import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf


def compute_best_values(table, metric_order):
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
        best_values[column_name] = (sorted_values[0], sorted_values[1])
    return best_values


def create_table_head(result_latex_code, caption, table_lable, cfg):
    used_columns = cfg.used_columns_dict[cfg.task]
    column_count = 0
    fars = [""]
    for key in used_columns:
        for column in used_columns[key]:
            if column == "models":
                continue
            far = column.split("=")[-1]
            fars.append(f"${far}$")
        column_count += len(used_columns[key]) - 1
    # column_count = (len(used_columns) - 1) * len(cfg.datasets)
    column_pretty_name = cfg.pretty_name.column
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
            result_latex_code += "& \\multicolumn{3}{c}{" + dataset_pretty_name + "} "
        result_latex_code += "\\\\\n"
        result_latex_code += "\\midrule\n"
    next_column_index = 1

    # for raw_column_name in used_columns:
    #     pretty_name = column_pretty_name[raw_column_name]
    #     if isinstance(pretty_name, str):
    #         pretty_name = [pretty_name]
    #     result_latex_code += (
    #         "\\begin{tabular}{c}\n" + "\\\\\n".join(pretty_name) + "\n\\end{tabular}"
    #     )
    #     if next_column_index < len(used_columns):
    #         result_latex_code += "&\n"
    #     else:
    #         result_latex_code += "\\\\\n"
    #         result_latex_code += "\midrule\n"
    #     next_column_index += 1
    return result_latex_code


def create_table_body(result_latex_code, cfg):
    dataset_to_metrics = {}
    dataset_to_best_values = {}
    for dataset in cfg.datasets:
        used_columns = cfg.used_columns = OmegaConf.to_container(cfg.used_columns_dict)[
            cfg.task
        ][dataset]
        table_path = cfg.metric_table_path.format(dataset=dataset)
        print(table_path)
        all_metric_values = pd.read_csv(table_path)
        all_metric_values = all_metric_values.drop(
            all_metric_values[
                (all_metric_values["models"] == "Random")
                | (all_metric_values["models"] == "Oracle")
            ].index
        )
        dataset_to_metrics[dataset] = all_metric_values
        dataset_to_best_values[dataset] = compute_best_values(
            all_metric_values[used_columns], cfg.metric_order
        )
    table = (
        dataset_to_metrics[cfg.datasets[0]]
        .set_index("models")
        .drop("Unnamed: 0", axis=1)
    )
    best_values = pd.DataFrame(dataset_to_best_values[cfg.datasets[0]])
    for i in range(len(cfg.datasets) - 1):
        # table = table.join(dataset_to_metrics[datasets[i+1]], on='models', how='left', lsuffix='_left', rsuffix='_right')
        next_table = dataset_to_metrics[cfg.datasets[i + 1]]
        next_table = next_table.set_index("models").drop("Unnamed: 0", axis=1)
        table = pd.concat([table, next_table], join="inner", axis=1)
        next_table_best = pd.DataFrame(dataset_to_best_values[cfg.datasets[i + 1]])
        best_values = pd.concat([best_values, next_table_best], join="inner", axis=1)
    table = table.reset_index()
    for row_index, (_, row) in enumerate(table.iterrows()):
        for column_index, column_name in enumerate(table.columns):
            if column_name == "models":
                result_latex_code += cfg.pretty_name.model[row[column_name]] + " & "
            elif column_name != "models":
                # metric value
                metric_value = row.iloc[column_index]
                if metric_value == best_values.iloc[0][column_index - 1]:
                    # best value
                    result_latex_code += (
                        "\\textbf{" + str(np.round(metric_value, cfg.round_num)) + "} "
                    )
                elif metric_value == best_values.iloc[1][column_index - 1]:
                    # second best value
                    result_latex_code += (
                        "\\underline{"
                        + str(np.round(metric_value, cfg.round_num))
                        + "} "
                    )
                else:
                    result_latex_code += (
                        f" {str(np.round(metric_value, cfg.round_num))} "
                    )
                if column_index < len(table.columns) - 1:
                    result_latex_code += "& "
                else:
                    # end of row
                    result_latex_code += "\\\\\n"
    return result_latex_code


def create_table_tail(result_latex_code, caption, table_lable, cfg):
    result_latex_code += "\\bottomrule\n"
    result_latex_code += "\\end{tabular}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\end{adjustbox}\n"
    # result_latex_code += "\\caption{" + caption + "}\n"
    # result_latex_code += "\\label{" + table_lable + "}\n"
    result_latex_code += "\\end{table}\n"

    return result_latex_code


@hydra.main(
    config_path="/app/configs/latex_tables",
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def run(cfg):
    result_latex_code = """"""
    if "{dataset_name}" in cfg.caption:
        caption = cfg.caption.format(
            dataset_name=cfg.datasets, task=cfg.pretty_name.task[cfg.task]
        )
    else:
        caption = cfg.caption

    # cfg.used_columns = OmegaConf.to_container(cfg.used_columns_dict)[cfg.task][
    #     cfg.datasets
    # ]
    # table head
    result_latex_code = create_table_head(
        result_latex_code,
        caption,
        cfg.table_lable,
        cfg,
    )

    # table body
    result_latex_code = create_table_body(result_latex_code, cfg)

    # table tail
    result_latex_code = create_table_tail(
        result_latex_code, caption, cfg.table_lable, cfg
    )

    # save result
    with open(Path(cfg.exp_dir) / "table.tex", "w") as fd:
        fd.write(result_latex_code)
    print("Out file:")
    print(str(Path(cfg.exp_dir) / "table.tex"))


if __name__ == "__main__":
    run()
