import hydra
import numpy as np
import pandas as pd

from typing import Dict, List
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from hydra.utils import instantiate

from rlplot import metrics
from rlplot import library as rly
from rlplot import plot_utils
from rlplot.plot_helpers import \
    load_all_exp_data, generate_intervals, generate_pairs, \
    save_metric, get_metric, read_and_norm_algo_scores, save_fig, \
    get_rank_matrix


REPS = 1000
CONFIDENCE = 0.68


def random_score_norm_func(task: str, scores: List):
    random_score = {
        'HalfCheetah-v4': -290.0479832104089,
        'Ant-v4': -55.14243068976598,
        'Walker2d-v4': 2.5912887180069686,
        'Humanoid-v4': 120.45141735893694
    }
    scores = np.array(scores)
    nume = scores - random_score[task]
    deno = np.max(scores) - random_score[task]
    return nume / deno


def create_diagnosis(
    n_epoch: int = 200,
    epoch_len: int = 5000,
    milestone_gap: int = 100000,
    metric: str = 'AverageEvalReward',
    algos: List = None, tasks: List = None,
    exp_dir: str = None,
    diagnosis_dir: str = None,
    **kwargs
):
    df: pd.DataFrame = \
        load_all_exp_data(exp_dir, algos, tasks)

    intervals: Dict = generate_intervals(
        gap=milestone_gap,
        epoch=n_epoch,
        epoch_len=epoch_len
    )

    exp_metric_dict = get_metric(df, metric, index='Exp')
    save_metric(exp_metric_dict, diagnosis_dir, intervals)

    algo_task_metric_dict = get_metric(df, metric, index='Algo')
    save_metric(algo_task_metric_dict, diagnosis_dir, intervals)

    task_algo_metric_dict = get_metric(df, metric, index='Task')
    save_metric(task_algo_metric_dict, diagnosis_dir, intervals)


def metric_curve(
    n_epoch: int = 200,
    epoch_len: int = 5000,
    metric: str = 'AverageEvalReward',
    aggregate_name: str = 'IQM',
    smooth_size: int = 1,
    algos: List = None, tasks: List = None,
    exp_dir: str = None,
    diagnosis_dir: str = None,
    fig_dir: str = None,
    **kwargs,
):

    def IQM(metric_val): return \
        np.array([metrics.aggregate_iqm(metric_val[..., step])
                  for step in range(metric_val.shape[-1])])

    def MEAN(metric_val): return \
        np.array([metrics.aggregate_mean(metric_val[..., step])
                  for step in range(metric_val.shape[-1])])

    def MEDIAN(metric_val): return \
        np.array([metrics.aggregate_median(metric_val[..., step])
                  for step in range(metric_val.shape[-1])])

    def OG(metric_val): return \
        np.array([metrics.aggregate_optimality_gap(metric_val[..., step], 1.0)
                  for step in range(metric_val.shape[-1])])

    aggregate_func_mapper = \
        {'Mean': MEAN, 'IQM': IQM, 'Median': MEDIAN, 'OG': OG}
    aggregate_func = aggregate_func_mapper[aggregate_name]

    df: pd.DataFrame = \
        load_all_exp_data(exp_dir, algos, tasks)

    xaxis = np.arange(1, n_epoch + 1) * epoch_len
    metric_dict = get_metric(df, metric, 'Task')

    for task, algo_dict in metric_dict.items():
        scores, cis = rly.get_interval_estimates(
            algo_dict, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE,
        )
        fig, ax = plot_utils.plot_metric_curve(
            xaxis, scores, cis,
            ylabel=f'{aggregate_name} {metric}',
            algorithms=algos,
            task=task,
            smooth_size=smooth_size,
        )
        save_fig(fig, f'metric_curve_{task.lower()}', fig_dir)

    algo_scores, normalized_algo_scores = \
        read_and_norm_algo_scores(
            diagnosis_dir, algos,
            milestone='all',
            norm_func=random_score_norm_func
        )

    scores, cis = \
        rly.get_interval_estimates(
            normalized_algo_scores, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE
        )

    fig, ax = plot_utils.plot_metric_curve(
        xaxis, scores, cis,
        ylabel=f'{aggregate_name} Normalized Score',
        algorithms=algos,
        task=tasks,
        smooth_size=smooth_size,
    )
    save_fig(fig, 'metric_curve', fig_dir)


def metric_value(
    milestone: str = '1m',
    aggregate_names: List[str] = ['Mean', 'IQM'],
    algos: List = None,
    diagnosis_dir: str = None,
    fig_dir: str = None,
    **kwargs,
):

    algo_scores, normalized_algo_scores = \
        read_and_norm_algo_scores(
            diagnosis_dir, algos, milestone,
            norm_func=random_score_norm_func,
        )

    aggregate_func_mapper = {
        'Mean': metrics.aggregate_mean,
        'IQM': metrics.aggregate_iqm,
        'Median': metrics.aggregate_median,
        'OG': metrics.aggregate_optimality_gap,
    }

    def aggregate_func(x): return \
        np.array([aggregate_func_mapper[name](x) for name in aggregate_names])

    aggregate_scores, aggregate_score_cis = \
        rly.get_interval_estimates(
            normalized_algo_scores, aggregate_func,
            reps=REPS, confidence_interval_size=CONFIDENCE
        )

    fig, axes = plot_utils.plot_metric_value(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=aggregate_names,
        algorithms=algos,
        milestone=milestone,
        xlabel='Normalized Score',
        xlabel_y_coordinate=-0.25,
    )

    save_fig(fig, 'metric_value', fig_dir)


def performance_profiles(
    tau: DictConfig,
    milestones: List[str] = ['300k', '500k', '1m'],
    algos: List = None,
    diagnosis_dir: str = None, fig_dir: str = None,
    **kwargs,
):
    tau = np.linspace(tau.low, tau.high, tau.nums)

    fig, axes = \
        plt.subplots(
            nrows=1, sharey=True, ncols=len(milestones),
            figsize=(3.5 * len(milestones), 2.5)
        )

    for i, milestone in enumerate(milestones):

        algo_scores, normalized_algo_scores = \
            read_and_norm_algo_scores(
                diagnosis_dir, algos, milestone,
                norm_func=random_score_norm_func
            )

        perf_prof, perf_prof_cis = \
            rly.create_performance_profile(
                normalized_algo_scores, tau,
                reps=REPS, confidence_interval_size=CONFIDENCE
            )

        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        plot_utils.plot_performance_profiles(
            perf_prof, tau,
            performance_profile_cis=perf_prof_cis,
            xlabel=r'Normalized Score $(\tau)$',
            ylabel=r'Fraction of runs with score $> \tau$' if i == 0 else None,
            labelsize='large',
            ticklabelsize='large',
            legend=False,
            alpha=0.1,
            ax=ax)
        ax.set_title(f'{milestone} steps', size='large')

        if i + 1 == len(milestones):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles[::-1], labels[::-1],
                loc='best', ncol=1,
                frameon=False, handlelength=1,
                borderaxespad=-1.,
            )
    fig.subplots_adjust(wspace=0.1)
    save_fig(fig, 'performance_profiles', fig_dir)


def probability_of_improvement(
    milestone: str = '1m',
    algos: List = None,
    diagnosis_dir: str = None,
    fig_dir: str = None,
    **kwargs,
):
    algo_scores, normalized_algo_scores = \
        read_and_norm_algo_scores(
            diagnosis_dir, algos, milestone,
            norm_func=random_score_norm_func
        )

    pairs = generate_pairs(algos)

    algo_score_pairs = {}
    for pair in pairs:
        d1 = normalized_algo_scores[pair[0]]
        d2 = normalized_algo_scores[pair[1]]
        algo_score_pairs[','.join(pair)] = (d1, d2)

    probabilities, probability_cis = rly.get_interval_estimates(
        algo_score_pairs, metrics.probability_of_improvement,
        reps=REPS, confidence_interval_size=CONFIDENCE
    )

    fig, ax = plot_utils.plot_probability_of_improvement(
        probabilities, probability_cis
    )
    save_fig(fig, "probability_of_improvement", fig_dir)


def sample_efficiency_curve(
    steps: List[int],
    epoch_len: int = 5000,
    algos: List = None,
    diagnosis_dir: str = None,
    fig_dir: str = None,
    **kwargs,
):
    algo_scores, normalized_algo_scores = \
        read_and_norm_algo_scores(
            diagnosis_dir, algos, 'all',
            norm_func=random_score_norm_func
        )

    steps = np.array(steps) - 1
    normalized_algo_steps_scores_dict = {algo: scores[:, :, steps] for algo, scores
                                         in normalized_algo_scores.items()}
    def IQM(scores): return np.array([metrics.aggregate_iqm(scores[..., frame])
                                      for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        normalized_algo_steps_scores_dict, IQM,
        reps=REPS, confidence_interval_size=CONFIDENCE
    )
    fig, ax = plot_utils.plot_sample_efficiency_curve(
        (steps + 1) * epoch_len / 1e6, iqm_scores, iqm_cis,
        algorithms=algos,
        xlabel=r'Number of Steps (in millions)',
        ylabel='IQM Normalized Score',
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        loc='upper left', ncol=1,
        frameon=False, handlelength=1,
        borderaxespad=-1.,
    )
    save_fig(fig, "sample_efficiency_curve", fig_dir)


def overall_ranks(
    milestones: List[str] = ['300k', '500k', '1m'],
    algos: List = None,
    diagnosis_dir: str = None,
    fig_dir: str = None,
    **kwargs,
):
    mean_ranks_all = {}
    all_ranks_individual = {}

    fig, axes = \
        plt.subplots(
            nrows=1, sharey=True, ncols=len(milestones),
            figsize=(3.5 * len(milestones), 2.5)
        )

    for i, milestone in enumerate(milestones):
        algo_scores, normalized_algo_scores = \
            read_and_norm_algo_scores(
                diagnosis_dir, algos, milestone,
                norm_func=random_score_norm_func
            )

        # num_task * (num_algo * num_algo)
        rank_matrix = \
            get_rank_matrix(normalized_algo_scores, n=50, algorithms=algos)
        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        ylabel = 'Fraction (in %)' if i == 0 else None
        legend = True if i + 1 == len(milestones) else False
        _, ax = plot_utils.plot_overall_ranks(
            rank_matrix,
            algorithms=algos,
            ylabel=ylabel,
            labelsize='large',
            ticklabelsize='medium',
            legend=legend,
            ax=ax,
        )
        ax.set_title(milestone + ' steps', size='large', y=0.95)
    fig.subplots_adjust(top=0.78, wspace=0.1, hspace=0.05)
    save_fig(fig, "overall_ranks", fig_dir)


@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg: DictConfig):
    instantiate(cfg)


if __name__ == '__main__':
    main()
