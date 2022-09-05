import numpy as np

from scipy.stats import stats

from analyse_result import load_result_dict, summarise_result
from utils.train_eval_utils import get_parser


def compute_cross_ins_p_value(model):
    args = get_parser()
    args.model = model
    args.seg, args.prior, args.align = True, True, True
    p_value_list = []

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
        diff_ins_list = [
            max([ins_dict[ins] for ins in range(1, 8) if ins != query_ins])
            for _, name_dict in result_dict.items()
            for _, ins_dict in name_dict.items()
        ]
        p_value = stats.ttest_rel(diff_ins_list, same_ins_list)[1]
        p_value_list.append(p_value)
    print(p_value_list)
    return p_value_list


def compute_novel_base_p_value():
    args = get_parser()
    args.query_ins = 3
    # args.model = "baseline_2d"
    args.seg, args.prior, args.align = True, True, True
    result_dict = {}

    for fold in range(1, 8):
        args.fold = fold
        result_dict.update(
            load_result_dict(args, "dice")
        )

    # {class: {name: {ins: v}}}
    novel_list = [
        ins_dict[3]
        for _, name_dict in result_dict.items()
        for _, ins_dict in name_dict.items()
    ]
    base_list = [
        np.mean(
            np.array([ins_dict[ins] for ins in range(1, 8) if ins != 3])
        )
        for _, name_dict in result_dict.items()
        for _, ins_dict in name_dict.items()
    ]
    p_value = stats.ttest_rel(base_list, novel_list)
    print(p_value)


def calculate_improvement(ins):
    args = get_parser()
    args.seg, args.prior, args.align = True, True, True
    args.model = "baseline_2d"
    args.novel_ins = ins
    args.query_ins = ins
    result = summarise_result(args, "dice", supervised=False)
    baseline_base, baseline_novel, baseline_all = result["mean"]["base"], result["mean"]["novel"], result["mean"]["all"]
    args.model = "ours"
    result = summarise_result(args, "dice", supervised=False)
    ours_base, ours_novel, ours_all = result["mean"]["base"], result["mean"]["novel"], result["mean"]["all"]
    print(f"ins{ins} all improvement: {ours_all - baseline_all}")
    print(f"ins{ins} base improvement: {ours_base - baseline_base}")
    print(f"ins{ins} novel improvement: {ours_novel - baseline_novel}")


if __name__ == '__main__':
    # compute_cross_ins_p_value("baseline_2d")
    # compute_cross_ins_p_value("ours")
    # compute_novel_base_p_value()
    calculate_improvement(ins=3)
    calculate_improvement(ins=4)