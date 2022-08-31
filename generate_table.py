import itertools
import os
import numpy as np

from pylatex import Tabular, MultiColumn, MultiRow
from pylatex.base_classes import Container
from pylatex.utils import bold, NoEscape

from analyse_result import summarise_result
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


def start_table(header, metric, by="fold"):
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
        table = Tabular('|c|c|' + 'ccccc|' * len(metric))
        table.add_hline()
        # table.add_row((
        #     MultiRow(2, data=header), MultiRow(2, data="s_ins"),
        #     *[MultiColumn(5, align='c|', data=metric_name_dict[m]) for m in metric]
        # ))
        # row = [
        #     "", "",
        #     *([*[f"fold{fold}" for fold in range(1, 5)], "mean"] * len(metric)),
        # ]
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
        table = Tabular('|c|c|' + 'cc|cc|cc|cc|c|' * len(metric))
        table.add_hline()
        # table.add_row((
        #     MultiRow(3, data=header), MultiRow(3, data="s_ins"),
        #     *[MultiColumn(9, align='c|', data=metric_name_dict[m]) for m in metric]
        # ))
        # table.add_hline(start=3)
        # row = [
        #     "", "",
        #     *([*[MultiColumn(2, align='c|', data=f"fold{fold}") for fold in range(1, 5)], "mean"] * len(metric))
        # ]
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
    args = get_parser()
    args.novel_ins = 3
    args.model = model
    args.con, args.align = True, True

    table = Tabular('|cc|ccccccc|cc|')
    table.add_hline()
    table.add_row((
        "", "", MultiColumn(7, align='c|', data='s_ins'), "std", "mean"
    ))
    table.add_row((
        "", "", "ins1", "ins2", "ins3", "ins4", "ins5", "ins6", "ins7", "", ""
    ))
    table.add_hline()

    for query_ins in range(1, 8):
        args.query_ins = query_ins
        result = summarise_result(args, metric="dice")  # {cls: {ins: v}}
        row = [result["mean"][support_ins] for support_ins in range(1, 8)]  # 7
        row = [i * 100 for i in row]
        # add std
        row = [*row, np.std(np.array(row)), np.mean(np.array(row))]
        # bold row maximum
        max = np.max(np.array(row))
        row = ["{:.2f}".format(i) if i != max else bold("{:.2f}".format(i)) for i in row]
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
    doc.generate_tex(f"./table/{model}_cross_ins_{metric}")


class Table(Container):

    def dumps(self):
        content = self.dumps_content()
        return content


if __name__ == '__main__':
    if not os.path.exists("./table"):
        os.mkdir("table")

    by = "fold"
    # by = "class"
    for metric in [["dice", "hausdorff"], ["dice"]]:
        get_result_table(ins=3, metric=metric, by=by)
        get_result_table(ins=4, metric=metric, by=by)
        get_training_size_ablation_table(metric=metric, by=by)
        get_k_shot_ablation_table(metric=metric, by=by)
    for model in ["ours", "baseline_2d"]:
        get_ins_correlation(model)