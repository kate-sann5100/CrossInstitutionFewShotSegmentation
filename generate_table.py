import itertools
import os
import numpy as np

from pylatex import Tabular, MultiColumn, MultiRow
from pylatex.base_classes import Container
from pylatex.utils import bold, NoEscape
from scipy import stats

from analyse_result import summarise_result, load_result_dict, summarise_base_ins_result
from statstics import compute_cross_ins_p_value
from utils.train_eval_utils import get_parser


def get_result_table(ins, metric, by):
    args = get_parser()
    args.novel_ins = ins
    args.query_ins = args.novel_ins

    table = start_table(header=["model"], metric=metric, by=by)
    args.model = "finetune"
    add_exp(args, table, exp_name="3d_finetune", metric=metric, by=by)
    args.model = "baseline_2d"
    add_exp(args, table, exp_name="2d", metric=metric, by=by)
    args.model = "ours"
    if args.novel_ins == 3:
        args.con, args.align = False, False
        add_exp(args, table, exp_name="3d", metric=metric, by=by)
        args.con, args.align = True, False
        add_exp(args, table, exp_name="3d_con", metric=metric, by=by)
        args.con, args.align = False, True
        add_exp(args, table, exp_name="3d_align", metric=metric, by=by)
    args.con, args.align = True, True
    add_exp(args, table, exp_name="3d_con_align", metric=metric, by=by)
    add_exp(args, table, exp_name="3d_supervised", metric=metric, by=by, supervised=True)
    doc = Table(data=table)
    doc.generate_tex(f"./table/ins{args.novel_ins}_{metric}")


def get_base_ins_result_table(ins, metric, by, base_only=True):
    args = get_parser()
    args.novel_ins = ins
    args.query_ins = args.novel_ins

    table = start_table(header=["model"], metric=metric, by=by, base_only=base_only)
    # args.model = "finetune"
    # add_exp(args, table, exp_name="3d_finetune", metric=metric, by=by)
    args.model = "baseline_2d"
    add_base_ins_exp(args, table, exp_name="2d", metric=metric, by=by, base_only=base_only)
    args.model = "ours"
    args.con, args.align = True, True
    add_base_ins_exp(args, table, exp_name="3d_con_align", metric=metric, by=by, base_only=base_only)
    # add_exp(args, table, exp_name="3d_supervised", metric=metric, by=by, supervised=True)
    doc = Table(data=table)
    doc.generate_tex(f"./table/base_ins_ins{args.novel_ins}_{metric}")


def add_base_ins_exp(args, table, exp_name, metric, by="fold", supervised=False, base_only=False):
    cat_list = ["base"] if base_only else ["all", "base", "novel"]
    metric_dict = {}
    for m in metric:
        result = summarise_base_ins_result(args, m, supervised=supervised)
        print("summarised")
        row_dict = {}
        for cat in cat_list:
            if by == "fold":
                row = np.array([
                    *[result[f"fold{fold}"][cat] for fold in range(1, 5)], result["mean"][cat]
                ])
            elif by == "class":
                row = np.array([
                    *[result[cls][cat] for cls in [1, 5, 2, 6, 3, 7, 4, 8]], result["mean"][cat]
                ])
            else:
                raise ValueError(f"unrecognised by: {by}")

            # convert dice to % value
            if m == "dice":
                row = row * 100
            row_dict[cat] = row

        if not base_only:
            row_dict[NoEscape('$\Delta$')] = (row_dict["novel"] - row_dict["base"]) / row_dict["novel"] * 100

        metric_dict[m] = row_dict

    column_list = cat_list if base_only else cat_list + [NoEscape('$\Delta$')]
    for i, cat in enumerate(column_list):
        row = list(
            itertools.chain.from_iterable(
                [metric_dict[m][cat] for m in metric]
            )
        )
        row = ["{:.2f}".format(i) for i in row]
        if cat == NoEscape('$\Delta$'):
            row = [f"{v}%" for v in row]
        if i == 0:
            if base_only:
                table.add_row(
                    (MultiRow(1, data=exp_name), *row)
                )
            else:
                table.add_row(
                    (MultiRow(1 if supervised else 4, data=exp_name), cat, *row)
                )
        else:
            if base_only:
                table.add_row(("", *row))
            else:
                table.add_row(("", cat, *row))

    table.add_hline()


def get_k_shot_ablation_table(metric, by):
    args = get_parser()
    args.novel_ins = 3
    args.query_ins = args.novel_ins
    args.model = "ours"
    args.con, args.align = True, True

    table = start_table(header=["# of shot"], metric=metric, by=by)
    for shot in range(1, 5):
        args.shot = shot
        add_exp(args, table, exp_name=shot, metric=metric, by=by)
    doc = Table(data=table)
    doc.generate_tex(f"./table/k_shot_ablation_{metric}")


def get_training_size_ablation_table(metric, by):
    args = get_parser()
    args.novel_ins = 3
    args.query_ins = args.novel_ins
    args.model = "ours"
    args.con, args.align = True, True

    table = start_table(header=["training", "data"], metric=metric, by=by)
    args.train_ratio = 1.
    add_exp(args, table, exp_name="whole", metric=metric, by=by)
    args.train_ratio = 0.5
    add_exp(args, table, exp_name="half", metric=metric, by=by)
    args.train_ratio = "ins1only"
    add_exp(args, table, exp_name="half_single_ins", metric=metric, by=by)
    args.train_ratio = 0.25
    add_exp(args, table, exp_name="quarter", metric=metric, by=by)
    doc = Table(data=table)
    doc.generate_tex(f"./table/training_size_ablation_{metric}")


def start_table(header, metric, by="fold", base_only=False):
    """
    Generate table head
    :param header:
    :param metric
    :return: table
    """
    metric_name_dict = {
        "dice": "Dice (%)",
        "hausdorff": "95% Hausdorff distance (mm)",
        "surface_distance": "Average Surface Distance (mm)"
    }
    if by == "fold":
        if base_only:
            table = Tabular('|c|' + 'ccccc|' * len(metric))
            table.add_hline()
            table.add_row((
                MultiRow(2, data=header[0]) if len(header) == 1 else header[0],
                *[MultiColumn(5, align='c|', data=metric_name_dict[m]) for m in metric]
            ))
            row = [
                "" if len(header) == 1 else header[1],
                *([*[f"fold{fold}" for fold in range(1, 5)], "mean"] * len(metric)),
            ]
        else:
            table = Tabular('|c|c|' + 'ccccc|' * len(metric))
            table.add_hline()
            table.add_row((
                MultiRow(2, data=header[0]) if len(header) == 1 else header[0], MultiRow(2, data="s_ins"),
                *[MultiColumn(5, align='c|', data=metric_name_dict[m]) for m in metric]
            ))
            row = [
                "" if len(header) == 1 else header[1], "",
                *([*[f"fold{fold}" for fold in range(1, 5)], "mean"] * len(metric)),
            ]
        table.add_row(row)
        table.add_hline()
        return table
    elif by == "class":
        if base_only:
            table = Tabular('|c|' + 'cc|cc|cc|cc|c|' * len(metric))
            table.add_hline()
            table.add_row((
                MultiRow(len(header), data=header[0]),
                *[MultiColumn(9, align='c|', data=metric_name_dict[m]) for m in metric]
            ))
            table.add_hline(start=3)
            row = [
                "",
                *([*[MultiColumn(2, align='c|', data=f"fold{fold}") for fold in range(1, 5)], "mean"] * len(metric))
            ]
            table.add_row(row)
        else:
            table = Tabular('|c|c|' + 'cc|cc|cc|cc|c|' * len(metric))
            table.add_hline()
            table.add_row((
                MultiRow(len(header), data=header[0]), MultiRow(4, data="s_ins"),
                *[MultiColumn(9, align='c|', data=metric_name_dict[m]) for m in metric]
            ))
            table.add_hline(start=3)
            row = [
                "", "",
                *([*[MultiColumn(2, align='c|', data=f"fold{fold}") for fold in range(1, 5)], "mean"] * len(metric))
            ]
            table.add_row(row)
        organ_list = [("bladder", ""), ("bone", ""), ("obturator", "internus"), ("transition", "zone"),
                      ("central", "gland"), ("rectum", ""), ("seminal", "vesicle"), ("neurovascular", "bundle")]
        for i in range(2):
            if base_only:
                row = [
                    MultiRow(2, data=header[1]) if i == 0 and len(header) == 2 else "",
                    *([
                          *[organ_list[c][i] for c in [0, 4, 1, 5, 2, 6, 3, 7]], ""
                      ] * len(metric))
                ]
            else:
                row = [
                    MultiRow(2, data=header[1]) if i == 0 and len(header) == 2 else "", "",
                    *([
                        *[organ_list[c][i] for c in [0, 4, 1, 5, 2, 6, 3, 7]], ""
                    ] * len(metric))
                ]
            table.add_row(row)
        table.add_hline()
        return table


def add_exp(args, table, exp_name, metric, by="fold", supervised=False):
    cat_list = ["N/A"] if supervised else ["all", "base", "novel"]
    metric_dict = {}
    for m in metric:
        result = summarise_result(args, m, supervised=supervised)
        print(result)
        row_dict = {}
        for cat in cat_list:
            if by == "fold":
                row = np.array([
                    *[result[f"fold{fold}"][cat] for fold in range(1, 5)], result["mean"][cat]
                ])
            elif by == "class":
                row = np.array([
                    *[result[cls][cat] for cls in [1, 5, 2, 6, 3, 7, 4, 8]], result["mean"][cat]
                ])
            else:
                raise ValueError(f"unrecognised by: {by}")

            # convert dice to % value
            if m == "dice":
                row = row * 100
            row_dict[cat] = row

        if not supervised:
            row_dict[NoEscape('$\Delta$')] = (row_dict["novel"] - row_dict["base"]) / row_dict["novel"] * 100

        metric_dict[m] = row_dict

    column_list = cat_list if supervised else cat_list + [NoEscape('$\Delta$')]
    for i, cat in enumerate(column_list):
        row = list(
            itertools.chain.from_iterable(
                [metric_dict[m][cat] for m in metric]
            )
        )
        row = ["{:.2f}".format(i) for i in row]
        if cat == NoEscape('$\Delta$'):
            row = [f"{v}%" for v in row]
        if i == 0:
            table.add_row((MultiRow(1 if supervised else 4, data=exp_name), cat, *row))
        else:
            table.add_row(("", cat, *row))

    table.add_hline()


def get_ins_correlation(model):
    """
    p-value refers to p-value from two tailed t-test performed per
    query institution between the Dice scores where support institution
    equals query institution and the max Dice scores achieved among
    all support institutions.
    :param model:
    :return:
    """
    args = get_parser()
    args.novel_ins = 3
    args.model = model
    args.con, args.align = True, True

    table = Tabular('|cc|ccccccc|ccc|')
    table.add_hline()
    table.add_row((
        "", "", MultiColumn(7, align='c|', data='s_ins'), "std", "mean", "p-value"
    ))
    table.add_row((
        "", "", "ins1", "ins2", "ins3", "ins4", "ins5", "ins6", "ins7", "", "", ""
    ))
    table.add_hline()
    p_value_list = compute_cross_ins_p_value(model)

    for query_ins in range(1, 8):
        args.query_ins = query_ins
        result = summarise_result(args, metric="dice")  # {cls: {ins: v}}
        row = [result["mean"][support_ins] for support_ins in range(1, 8)]  # 7
        row = [i * 100 for i in row]
        # add std
        row = [*row, np.std(np.array(row)), np.mean(np.array(row))]
        # bold row maximum
        print(row)
        max = np.max(np.array(row))
        row = ["{:.2f}".format(i) if i != max else bold("{:.2f}".format(i)) for i in row]
        p_value = p_value_list[query_ins - 1]
        row.append("{: .2e}".format(p_value) if p_value < 0.01 else "{: .2f}".format(p_value))
        if query_ins == 1:
            table.add_row((
                MultiRow(7, data='q_ins'),
                f"ins{query_ins}",
                *row
            ))
        else:
            table.add_row(("", f"ins{query_ins}", *row))

    table.add_hline()

    doc = Table(data=table)
    doc.generate_tex(f"./table/{model}_cross_ins_dice")


def get_cross_ins_p_value(model):
    """
    p-value from two tailed t-test performed per query institution
    between the Dice scores where support institution equals query
    institution and the Dice scores achieved per support
    institution.
    :param model:
    :return:
    """
    args = get_parser()
    args.novel_ins = 3
    args.model = model
    args.con, args.align = True, True

    table = Tabular('|cc|ccccccc|')
    table.add_hline()
    table.add_row((
        "", "", MultiColumn(7, align='c|', data='s_ins')
    ))
    table.add_row((
        "", "", "ins1", "ins2", "ins3", "ins4", "ins5", "ins6", "ins7"
    ))
    table.add_hline()

    for query_ins in range(1, 8):
        args.query_ins = query_ins
        result_dict = {}
        for fold in range(1, 5):
            args.fold = fold
            result_dict.update(
                load_result_dict(args, "dice")
            )

        # {class: {name: {ins: v}}}
        same_ins_list = [
            ins_dict[query_ins]
            for _, name_dict in result_dict.items()
            for _, ins_dict in name_dict.items()
        ]
        row = []
        for ins in range(1, 8):
            if ins == query_ins:
                p_value_string = "N/A"
            else:
                ins_list = [
                    ins_dict[ins]
                    for _, name_dict in result_dict.items()
                    for _, ins_dict in name_dict.items()
                ]
                p_value = stats.ttest_rel(ins_list, same_ins_list)[1]
                p_value_string = "{: .2e}".format(p_value) if p_value < 0.01 else "{: .2f}".format(p_value)
                if p_value > 0.05:
                    p_value_string = bold(p_value_string)
            row.append(p_value_string)
        if query_ins == 1:
            table.add_row((
                MultiRow(7, data='q_ins'),
                f"ins{query_ins}",
                *row
            ))
        else:
            table.add_row(("", f"ins{query_ins}", *row))

    table.add_hline()

    doc = Table(data=table)
    doc.generate_tex(f"./table/{model}_pvalue")


def get_p_value():
    """
    p-value from two tailed t-test performed per query institution
    except 5,6,7 between the Dice scores where support institution
    equals query institution and the max Dice scores achieved among
    all support institutions except 5,6,7 .
    :return:
    """
    args = get_parser()
    args.novel_ins = 3
    args.con, args.align = True, True

    table = Tabular('|c|cccc|')
    table.add_hline()
    table.add_row((
        MultiRow(2, data="model"), MultiColumn(4, align='c|', data='q_ins')
    ))
    table.add_row((
        "", "ins1", "ins2", "ins3", "ins4"
    ))
    table.add_hline()

    for model in ["ours", "baseline_2d"]:
        args.model = model
        row = []
        for query_ins in range(1, 5):
            args.query_ins = query_ins
            result_dict = {}
            for fold in range(1, 5):
                args.fold = fold
                result_dict.update(
                    load_result_dict(args, "dice")
                )

            # {class: {name: {ins: v}}}
            same_ins_list = [
                ins_dict[query_ins]
                for _, name_dict in result_dict.items()
                for _, ins_dict in name_dict.items()
            ]
            diff_ins_list = [
                max(np.array([ins_dict[ins] for ins in range(1, 5) if ins != query_ins]))
                for _, name_dict in result_dict.items()
                for _, ins_dict in name_dict.items()
            ]

            print(len(same_ins_list), len(diff_ins_list))
            print(np.array(same_ins_list) - np.array(diff_ins_list))
            p_value = stats.ttest_rel(same_ins_list, diff_ins_list)[1]
            print(p_value)
            p_value_string = "{: .2e}".format(p_value) if p_value < 0.01 else "{: .2f}".format(p_value)
            if p_value > 0.05:
                p_value_string = bold(p_value_string)
            row.append(p_value_string)

        table.add_row((model, *row))

    table.add_hline()

    doc = Table(data=table)
    doc.generate_tex(f"./table/pvalue")


class Table(Container):

    def dumps(self):
        content = self.dumps_content()
        return content


if __name__ == '__main__':
    if not os.path.exists("./table"):
        os.mkdir("table")

    by = "fold"
    for metric in [["dice", "hausdorff"], ["dice"]]:
        get_base_ins_result_table(ins=3, metric=metric, by=by)
    # by = "class"
    # for metric in [["dice", "hausdorff"], ["dice"]]:
    #     get_result_table(ins=3, metric=metric, by=by)
    #     get_result_table(ins=4, metric=metric, by=by)
    #     get_training_size_ablation_table(metric=metric, by=by)
    #     get_k_shot_ablation_table(metric=metric, by=by)
    # for model in ["ours", "baseline_2d"]:
        # get_ins_correlation(model)
        # get_cross_ins_p_value(model)
        # get_p_value()