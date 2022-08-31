import numpy as np
import itertools

from utils.train_eval_utils import get_parser, set_seed, get_save_dir


def summarise_result(args, metric="dice", supervised=False):

    if supervised:
        with open(f"ckpt/supervised/ins{args.novel_ins}/ins{args.novel_ins}_1shot_{metric}_result_dict.pkl", "rb") as fh:
            from pickle5 import pickle
            result = get_result(
                args, pickle.load(fh), metric=metric, supervised=True
            )

    else:
        # combine results for all 4 folds
        result = {}  # {cls: {ins: v}}
        for fold in range(1, 5):
            args.fold = fold
            result_dict = load_result_dict(args, metric)
            result.update(
                get_result(args, result_dict)
            )

    cat_list = ["N/A"] if supervised else ["all", "base", "novel"]
    # average overall
    result["mean"] = {
        ins_cat: np.mean(np.array([result[cls][ins_cat] for cls in range(1, 9)]))
        for ins_cat in cat_list
    }

    if not supervised:
        result["mean"].update({
            ins: np.mean(np.array([result[cls][ins] for cls in range(1, 9)]))
            for ins in range(1, 8)
        })

    # average by fold
    for fold in range(1, 5):
        result[f"fold{fold}"] = {
            ins_cat: np.mean(
                np.array(
                    [result[cls][ins_cat]
                     for cls in [fold, fold + 4]]
                )
            )
            for ins_cat in cat_list
        }

    # print mean
    print(f"----------mean----------")
    for cat in cat_list:
        mean = result["mean"][cat]
        print(f"{cat}: {mean}")
    return result


def load_result_dict(args, metric):
    save_dir = get_save_dir(args)
    dict_name = f"ins{args.query_ins}_{args.shot}shot_{metric}_result_dict.pkl"
    with open(f"{save_dir}/{dict_name}", "rb") as fh:
        from pickle5 import pickle
        result_dict = pickle.load(fh)
    return result_dict


def get_result(args, result_dict, metric=None, supervised=False):
    """
    :param args:
    :param result_dict: {class: {name: {ins: v}}}
    :param metric: str
    :return: {cls: {ins_cat: v}}
    """
    novel_cls = list(range(1, 9)) if supervised else [args.fold, args.fold + 4]
    result = {cls: {} for cls in novel_cls}

    # average by cls
    ins_list = ["N/A"] if supervised else list(range(1, 8))
    for ins in ins_list:
        for cls in novel_cls:
            name_ins_result_dict = result_dict[cls]
            result_list = [
                ins_value_dict[ins]
                for name, ins_value_dict in name_ins_result_dict.items() if ins in ins_value_dict.keys()
            ]
            result_list = [r for r in result_list if not np.isinf(r)]
            result[cls][ins] = np.mean(result_list)

    # average by support ins category
    if not supervised:
        cat_ins_dict = {
            "base": [ins for ins in range(1, 8) if ins != args.novel_ins],
            "novel": [args.novel_ins],
            "all": list(range(1, 8))
        }
        for cat, ins_list in cat_ins_dict.items():
            for cls in novel_cls:
                name_ins_result_dict = result_dict[cls]
                result_list = [
                    [ins_value_dict[ins] for ins in ins_list if ins in ins_value_dict.keys()]
                    for name, ins_value_dict in name_ins_result_dict.items()
                ]
                result_list = list(itertools.chain.from_iterable(result_list))
                result_list = [r for r in result_list if not np.isinf(r)]
                result[cls][cat] = np.mean(result_list)

        if metric is not None:
            print(f"----------fold{args.fold} {metric}----------")
            for cat in ["all", "base", "novel"]:
                mean = np.mean(np.array([result[cls][cat] for cls in novel_cls]))
                print(f"{cat}: {mean}")
    return result


if __name__ == '__main__':
    args = get_parser()
    args.novel_ins = 3
    args.query_ins = args.novel_ins
    args.fold = 3
    args.shot = 5
    result_dict = load_result_dict(args, metric="dice")
    get_result(args, result_dict, metric="dice", supervised=False)