import numpy as np
import pandas as pd
import inspect
from copy import deepcopy
from typing import Callable, List, Dict, Union
from pathlib import Path
from omegaconf import OmegaConf
from itertools import product
from collections import defaultdict, Counter


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def logging(func):
    def wrapper(*args, **kwargs):
        print()
        print('==============' * 8)

        if getattr(func, '__name__') == 'plot_metric_curve':
            default_kwargs = _get_default_args(func)
            kwargs = {**default_kwargs, **kwargs}
            print(
                'Plotting metric curve:\n',
                f' xlabel = {kwargs.get("xlabel")}\n',
                f' ylabel = {kwargs.get("ylabel")}\n',
                f' algorithms = {kwargs.get("algorithms")}\n',
                f' task = {kwargs.pop("task", None)}\n',
            )
            return func(*args, **kwargs)

        elif getattr(func, '__name__') == 'plot_metric_value':
            default_kwargs = _get_default_args(func)
            kwargs = {**default_kwargs, **kwargs}
            print(
                'Plotting metric value:\n',
                f' xlabel = {kwargs.get("xlabel")}\n',
                f' algorithms = {kwargs.get("algorithms")}\n',
                f' metric names = {kwargs.get("metric_names")}\n',
                f' milestone = {kwargs.get("milestone")}\n',
            )
            return func(*args, **kwargs)

        elif getattr(func, '__name__') == 'plot_performance_profiles':
            default_kwargs = _get_default_args(func)
            kwargs = {**default_kwargs, **kwargs}
            print(
                'Plotting performance profiles:\n',
                f' xlabel = {kwargs.get("xlabel")}\n',
                f' algorithms = {list(args[0].keys())}\n',
            )
            return func(*args, **kwargs)

        elif getattr(func, '__name__') == 'plot_probability_of_improvement':
            default_kwargs = _get_default_args(func)
            kwargs = {**default_kwargs, **kwargs}
            print(
                'Plotting probability of improvement:\n',
                f' xlabel = {kwargs.get("xlabel")}\n',
                f' pairs = {list(args[0].keys())}\n',
            )
            return func(*args, **kwargs)

        elif getattr(func, '__name__') == 'plot_overall_ranks':
            default_kwargs = _get_default_args(func)
            kwargs = {**default_kwargs, **kwargs}
            print(
                'Plotting overall ranks:\n',
                f' algorithms = {kwargs.get("algorithms")}\n',
            )
            return func(*args, **kwargs)

        elif getattr(func, '__name__') == 'save_fig':
            file_path = func(*args, **kwargs)
            print(f'Saving figure --> {file_path}')
            return file_path

        elif getattr(func, '__name__') == 'save_yaml':
            file_path = func(*args, **kwargs)
            print(f'Saving yaml file --> {file_path}')
            return file_path

    return wrapper


def load_exp_data(
    exp_dir: str,
    data_name: str = 'progress.csv',
    load_func: Callable = lambda path: pd.read_csv(path, sep=','),
    drop_na=False,
) -> pd.DataFrame:
    exp_dir, exp_data = Path(exp_dir), pd.DataFrame()
    assert exp_dir.exists(), f'Cannot find exp: [{exp_dir}], Pls check!'
    for run_path in exp_dir.rglob(data_name):
        run_dir = run_path.parent
        run_data = load_func(run_path)
        run_data['Exp'], run_data['Run'] = exp_dir.name, run_dir.name
        run_data['Algo'], run_data['Task'] = exp_dir.name.split('_')
        exp_data = exp_data.append(run_data, ignore_index=True)
    return exp_data.dropna(axis=1) if drop_na else exp_data


def load_all_exp_data(
    exp_dir: str,
    algos: List,
    tasks: List,
    **kwargs,
) -> pd.DataFrame:
    exp_dir = Path(exp_dir)
    all_exp_data = pd.DataFrame()
    for algo, task in product(algos, tasks):
        exp_path = exp_dir / f'{algo}_{task}'
        all_exp_data = all_exp_data.append(load_exp_data(exp_path, **kwargs), ignore_index=True)
    return all_exp_data.dropna(axis=1) if kwargs.get('drop_na', False) else all_exp_data


def get_axis_kwargs(**kwargs):
    axis_kwargs = dict(
        labelsize=kwargs.get('labelsize', 'x-large'),
        ticklabelsize=kwargs.get('ticklabelsize', 'x-large'),
        xticks=kwargs.get('xticks', None),
        xticklabels=kwargs.get('xticklabels', None),
        yticks=kwargs.get('yticks', None),
        legend=kwargs.get('legend', False),
        grid_alpha=kwargs.get('grid_alpha', 0.2),
        legendsize=kwargs.get('legendsize', 'x-large'),
        xlabel=kwargs.get('xlabel', ''),
        ylabel=kwargs.get('ylabel', ''),
        wrect=kwargs.get('wrect', 10),
        hrect=kwargs.get('hrect', 10),
    )
    return axis_kwargs


def get_metric(
    df: pd.DataFrame,
    metric: str = 'AverageEvalReward',
    index: str = 'Exp'
) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:

    index = index.capitalize()
    assert index in ('Exp', 'Algo', 'Task')
    exp_names, algo_names, task_names = \
        list(df['Exp'].unique()), list(df['Algo'].unique()), list(df['Task'].unique())
    metric_vals = {}
    for exp_name in exp_names:
        exp_df = df[df['Exp'] == exp_name]
        arr = exp_df.groupby('Run')[metric].apply(list).values
        metric_vals[exp_name] = align_and_stack(arr)
    if index == 'Exp':
        return metric_vals
    elif index == 'Algo':
        nested_metric_vals = defaultdict(dict)
        for algo, task in product(algo_names, task_names):
            nested_metric_vals[algo][task] = metric_vals[f'{algo}_{task}']
    else:
        nested_metric_vals = defaultdict(dict)
        for task, algo in product(task_names, algo_names):
            nested_metric_vals[task][algo] = metric_vals[f'{algo}_{task}']
    return nested_metric_vals


def save_metric(
    metric_dict: Union[Dict[str, Dict[str, List[List]]], Dict[str, List[List]]],
    save_dir: str, intervals: Dict,
):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def func(interval, val: List[List]):
        ll, rr = max(0, interval - 2), min(len(val[0]), interval + 2)
        return np.mean([v[ll - 1:rr] for v in val], axis=1).tolist()

    # save itr scores (100k as interval)
    for name, val in metric_dict.items():
        itr_dict = {}
        for key, interval in intervals.items():
            if isinstance(val, dict):
                itr_dict[key] = {k: func(interval, v) for k, v in val.items()}
            elif isinstance(val, (list, np.ndarray)):
                itr_dict[key] = func(interval, val)
            else:
                raise TypeError

        # save all metric val
        if isinstance(val, dict):
            itr_dict['all'] = {k: v.tolist() for k, v in val.items()}
        elif isinstance(val, list):
            itr_dict['all'] = val
        elif isinstance(val, np.ndarray):
            itr_dict['all'] = val.tolist()
        else:
            raise TypeError

        save_yaml(itr_dict, name=name, dir=save_dir)


@logging
def save_fig(fig, name, dir=None):
    if dir is None:
        dir = Path.cwd()
    Path(dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{name}.pdf'
    file_path = Path(dir) / f'{name}.pdf'
    fig.savefig(file_path, format='pdf', bbox_inches='tight')
    return file_path


@logging
def save_yaml(
    val: Dict[str, Union[np.ndarray, List]],
    name: str, dir: str
):
    if dir is None:
        dir = Path.cwd()
    file_path = (Path(dir) / name).with_suffix('.yaml')
    OmegaConf.save(val, file_path)
    return file_path


def save_figs(figs, names, dir=None):
    file_names = [save_fig(fig, name, dir) for fig, name in zip(figs, names)]
    return file_names


def save_yamls(vals, names, dir=None):
    file_names = [save_yaml(val, name, dir) for val, name in zip(vals, names)]
    return file_names


def align_and_stack(xs):
    lens = [len(x) for x in xs]
    min_len = min(lens)
    return np.stack([x[:min_len] for x in xs], axis=0)


def generate_intervals(gap=100000, epoch=200, epoch_len=5000):
    intervals = {}
    for point in range(gap, epoch * epoch_len + gap, gap):
        if point >= 1e6:
            key = str(point // 1000000) + 'm'
        else:
            key = str(point // 1000) + 'k'
        val = point // epoch_len
        intervals[key] = val
    return intervals


def generate_pairs(elements):
    pairs = []
    elements = elements[::-1]
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            pairs.append([elements[i], elements[j]])
    return pairs[::-1]


def convert_to_matrix(score_dict, sort=False):
    if sort:
        keys = sorted(list(score_dict.keys()))
    else:
        keys = list(score_dict.keys())
    return np.stack([score_dict[k] for k in keys], axis=1)


def read_milestone_from_yaml(
    dir, name,
    milestone='1m'
):
    file_path = (Path(dir) / name).with_suffix('.yaml')
    out = OmegaConf.load(file_path)
    out = out[milestone]
    return out


def read_and_norm_algo_scores(
    dir, algos, milestone: str,
    norm_func: Callable,
):
    algo_scores = {
        algo: read_milestone_from_yaml(dir, algo, milestone)
        for algo in algos
    }
    normalized_algo_scores = deepcopy(algo_scores)
    for algo in normalized_algo_scores:
        normalized_algo_scores[algo] = \
            {task: norm_func(task, scores)
                for task, scores in normalized_algo_scores[algo].items()}
    for algo, task_scores in algo_scores.items():
        for task, scores in task_scores.items():
            assert np.argmax(algo_scores[algo][task]) \
                == np.argmax(normalized_algo_scores[algo][task])

    # num_runs * num_tasks
    algo_scores = \
        {algo: convert_to_matrix(scores) for algo, scores in algo_scores.items()}
    normalized_algo_scores = \
        {algo: convert_to_matrix(scores) for algo, scores in normalized_algo_scores.items()}

    return algo_scores, normalized_algo_scores


def subsample_scores_mat(score_mat, num_samples=5, replace=False):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    subsampled_scores = np.empty((num_samples, num_games))
    for i in range(num_games):
        indices = np.random.choice(total_samples, size=num_samples, replace=replace)
        subsampled_scores[:, i] = score_mat[indices, i]
    return subsampled_scores


def get_rank_matrix(score_dict, n=10000, algorithms=None):
    arr = []
    if algorithms is None:
        algorithms = sorted(score_dict.keys())
    for alg in algorithms:
        arr.append(subsample_scores_mat(
            score_dict[alg], num_samples=n, replace=True))
    X = np.stack(arr, axis=0)
    num_algs, _, num_tasks = X.shape
    all_mat = []
    for task in range(num_tasks):
        # Sort based on negative scores as rank 0 corresponds to minimum value,
        # rank 1 corresponds to second minimum value when using lexsort.
        task_x = -X[:, :, task]
        # This is done to randomly break ties.
        rand_x = np.random.random(size=task_x.shape)
        # Last key is the primary key,
        indices = np.lexsort((rand_x, task_x), axis=0)
        mat = np.zeros((num_algs, num_algs))
        for rank in range(num_algs):
            cnts = Counter(indices[rank])
            mat[:, rank] = np.array([cnts[i] / n for i in range(num_algs)])
        all_mat.append(mat)
    all_mat = np.stack(all_mat, axis=0)
    return all_mat
