import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from rlplot.plot_helpers import logging
from itertools import product

# yanked and modified from https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py

def _non_linear_scaling(performance_profiles,
                        tau_list,
                        xticklabels=None,
                        num_points=5,
                        log_base=2):
    """Returns non linearly scaled tau as well as corresponding xticks.
    The non-linear scaling of a certain range of threshold values is proportional
    to fraction of runs that lie within that range.
    Args:
      performance_profiles: A dictionary mapping a method to its performance
        profile, where each profile is computed using thresholds in `tau_list`.
      tau_list: List or 1D numpy array of threshold values on which the profile is
        evaluated.
      xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
      num_points: If `xticklabels` are not passed, then specifices the number of
        indices to be generated on a log scale.
      log_base: Base of the logarithm scale for non-linear scaling.
    Returns:
      nonlinear_tau: Non-linearly scaled threshold values.
      new_xticks: x-axis ticks from `nonlinear_tau` that would be plotted.
      xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
    """

    methods = list(performance_profiles.keys())
    nonlinear_tau = np.zeros_like(performance_profiles[methods[0]])
    for method in methods:
        nonlinear_tau += performance_profiles[method]
    nonlinear_tau /= len(methods)
    nonlinear_tau = 1 - nonlinear_tau

    if xticklabels is None:
        tau_indices = np.int32(
            np.logspace(
                start=0,
                stop=np.log2(len(tau_list) - 1),
                base=log_base,
                num=num_points))
        xticklabels = [tau_list[i] for i in tau_indices]
    else:
        tau_as_list = list(tau_list)
        # Find indices of x which are in `tau`
        tau_indices = [tau_as_list.index(x) for x in xticklabels]
    new_xticks = nonlinear_tau[tau_indices]
    return nonlinear_tau, new_xticks, xticklabels


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    return ax


def _annotate_and_decorate_axis(
    ax,
    labelsize='x-large',
    ticklabelsize='x-large',
    xticks=None,
    xticklabels=None,
    yticks=None,
    legend=True,
    grid_alpha=0.3,
    legendsize='x-large',
    xlabel='',
    ylabel='',
    wrect=10,
    hrect=10
):
    """Annotates and decorates the plot."""
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha, linewidth=2)
    ax = _decorate_axis(
        ax,
        wrect=wrect,
        hrect=hrect,
        ticklabelsize=ticklabelsize
    )
    if legend:
        # ax.legend(fontsize=legendsize)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],
            loc='best', ncol=1,
            frameon=False, handlelength=1,
            borderaxespad=0., fontsize=legendsize
        )
    return ax


@logging
def plot_metric_curve(
    frames,
    point_estimates,
    interval_estimates,
    algorithms=None,
    colors=None,
    color_palette=None,
    figsize=(7, 5),
    xlabel='TotalEnvInteracts',
    ylabel='AverageEvalReward',
    ax=None,
    labelsize='xx-large',
    ticklabelsize='xx-large',
    smooth_size=1,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))[::-1]
        colors = dict(zip(algorithms, color_palette))

    for algorithm in algorithms:
        metric_value = point_estimates[algorithm]
        lower, upper = interval_estimates[algorithm]
        if smooth_size > 1:
            metric_value = uniform_filter1d(metric_value, size=smooth_size)
            lower = uniform_filter1d(lower, size=smooth_size)
            upper = uniform_filter1d(upper, size=smooth_size)
        ax.plot(
            frames,
            metric_value,
            color=colors[algorithm],
            linewidth=kwargs.get('linewidth', 3),
            label=algorithm)
        ax.fill_between(
            frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.10)
    kwargs.pop('marker', '0')
    kwargs.pop('linewidth', '2')

    ax = _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs)
    return fig, ax


@logging
def plot_metric_value(
    point_estimates,
    interval_estimates,
    metric_names,
    algorithms=None,
    colors=None,
    color_palette=None,
    max_ticks=4,
    subfigure_width=3.4,
    row_height=0.43,
    xlabel_y_coordinate=-0.1,
    xlabel='Normalized Score',
    **kwargs
):
    """Plots various metrics with confidence intervals.
    Args:
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metrics to plot.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      metric_names: Names of the metrics corresponding to `point_estimates`.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
        `plt.MaxNLocator`.
      subfigure_width: Width of each subfigure.
      row_height: Height of each row in a subfigure.
      xlabel_y_coordinate: y-coordinate of the x-axis label.
      xlabel: Label for the x-axis.
      **kwargs: Arbitrary keyword arguments.
    Returns:
      fig: A matplotlib Figure.
      axes: `axes.Axes` or array of Axes.
    """

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])
    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))[::-1]
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop('interval_height', 0.6)

    for idx, metric_name in enumerate(metric_names):
        for alg_idx, algorithm in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # Plot interval estimates.
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=colors[algorithm],
                alpha=0.75,
                label=algorithm)
            # Plot point estimates.
            ax.vlines(
                x=point_estimates[algorithm][idx],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=algorithm,
                color='k',
                alpha=0.5)

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algorithms, fontsize='x-large')
        ax.set_title(metric_name, fontsize='xx-large')
        ax.tick_params(axis='both', which='major')
        _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)
    fig.text(0.4, xlabel_y_coordinate, xlabel, ha='center', fontsize='xx-large')
    plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
    return fig, axes


@logging
def plot_performance_profiles(
    performance_profiles,
    tau_list,
    performance_profile_cis=None,
    use_non_linear_scaling=False,
    ax=None,
    colors=None,
    color_palette=None,
    alpha=0.15,
    figsize=(10, 5),
    xticks=None,
    yticks=None,
    xlabel=r'Normalized Score ($\tau$)',
    ylabel=r'Fraction of runs with score $> \tau$',
    linestyles=None,
    **kwargs
):
    """Plots performance profiles with stratified confidence intervals.
    Args:
      performance_profiles: A dictionary mapping a method to its performance
        profile, where each profile is computed using thresholds in `tau_list`.
      tau_list: List or 1D numpy array of threshold values on which the profile is
        evaluated.
      performance_profile_cis: The confidence intervals (default 95%) of
        performance profiles evaluated at all threshdolds in `tau_list`.
      use_non_linear_scaling: Whether to scale the x-axis in proportion to the
        number of runs within any specified range.
      ax: `matplotlib.axes` object.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object. Used when `colors` is None.
      alpha: Changes the transparency of the shaded regions corresponding to the
        confidence intervals.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      xticks: The list of x-axis tick locations. Passing an empty list removes all
        xticks.
      yticks: The list of y-axis tick locations between 0 and 1. If None, defaults
        to `[0, 0.25, 0.5, 0.75, 1.0]`.
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      linestyles: Maps each method to a linestyle. If None, then the 'solid'
        linestyle is used for all methods.
      **kwargs: Arbitrary keyword arguments for annotating and decorating the
        figure. For valid arguments, refer to `_annotate_and_decorate_axis`.
    Returns:
      `matplotlib.axes.Axes` object used for plotting.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if colors is None:
        keys = performance_profiles.keys()
        color_palette = sns.color_palette(color_palette, n_colors=len(keys))[::-1]
        colors = dict(zip(list(keys), color_palette))

    if linestyles is None:
        linestyles = {key: 'solid' for key in performance_profiles.keys()}

    if use_non_linear_scaling:
        tau_list, xticks, xticklabels = _non_linear_scaling(performance_profiles,
                                                            tau_list, xticks)
    else:
        xticklabels = xticks

    for method, profile in performance_profiles.items():
        ax.plot(
            tau_list,
            profile,
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=kwargs.pop('linewidth', 2.0),
            label=method)
        if performance_profile_cis is not None:
            if method in performance_profile_cis:
                lower_ci, upper_ci = performance_profile_cis[method]
                ax.fill_between(
                    tau_list, lower_ci, upper_ci, color=colors[method], alpha=alpha)

    if yticks is None:
        yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax = _annotate_and_decorate_axis(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticklabels,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs
    )
    return fig, ax


@logging
def plot_probability_of_improvement(
        probability_estimates,
        probability_interval_estimates,
        pair_separator=',',
        ax=None,
        figsize=(4, 3),
        colors=None,
        color_palette=None,
        alpha=0.75,
        xticks=None,
        xlabel='P(X > Y)',
        left_ylabel='Algorithm X',
        right_ylabel='Algorithm Y',
        **kwargs):
    """Plots probability of improvement with confidence intervals.
    Args:
      probability_estimates: Dictionary mapping algorithm pairs (X, Y) to a
        list or array containing probability of improvement of X over Y.
      probability_interval_estimates: Dictionary mapping algorithm pairs (X, Y)
        to interval estimates corresponding to the `probability_estimates`.
        Typically, consists of stratified independent bootstrap CIs.
      pair_separator: Each algorithm pair name in dictionaries above is joined by
        a string separator. For example, if the pairs are specified as 'X;Y', then
        the separator corresponds to ';'. Defaults to ','.
      ax: `matplotlib.axes` object.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      colors: Maps each algorithm pair id to a color. If None, then this mapping
        is created based on `color_palette`.
      color_palette: `seaborn.color_palette` object. Used when `colors` is None.
      alpha: Changes the transparency of the shaded regions corresponding to the
        confidence intervals.
      xticks: The list of x-axis tick locations. Passing an empty list removes all
        xticks.
      xlabel: Label for the x-axis. Defaults to 'P(X > Y)'.
      left_ylabel: Label for the left y-axis. Defaults to 'Algorithm X'.
      right_ylabel: Label for the left y-axis. Defaults to 'Algorithm Y'.
      **kwargs: Arbitrary keyword arguments for annotating and decorating the
        figure. For valid arguments, refer to `_annotate_and_decorate_axis`.
    Returns:
      `axes.Axes` which contains the plot for probability of improvement.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    if not colors:
        colors = sns.color_palette(
            color_palette, n_colors=len(probability_estimates))[::-1]
    h = kwargs.pop('interval_height', 0.6)
    wrect = kwargs.pop('wrect', 5)
    ticklabelsize = kwargs.pop('ticklabelsize', 'x-large')
    labelsize = kwargs.pop('labelsize', 'x-large')
    # x-position of the y-label
    ylabel_x_coordinate = kwargs.pop('ylabel_x_coordinate', 0.2)
    # x-position of the y-label

    twin_ax = ax.twinx()
    all_algorithm_x, all_algorithm_y = [], []

    # Main plotting code
    for idx, (algorithm_pair, prob) in enumerate(probability_estimates.items()):
        lower, upper = probability_interval_estimates[algorithm_pair]
        algorithm_x, algorithm_y = algorithm_pair.split(pair_separator)
        all_algorithm_x.append(algorithm_x)
        all_algorithm_y.append(algorithm_y)

        ax.barh(
            y=idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[idx],
            alpha=alpha,
            label=algorithm_x)
        twin_ax.barh(
            y=idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[idx],
            alpha=0.0,
            label=algorithm_y)
        ax.vlines(
            x=prob,
            ymin=idx - 7.5 * h / 16,
            ymax=idx + (6 * h / 16),
            color='k',
            alpha=min(alpha + 0.1, 1.0))

    # Beautify plots
    yticks = range(len(probability_estimates))
    ax = _annotate_and_decorate_axis(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        ylabel=left_ylabel,
        wrect=wrect,
        ticklabelsize=ticklabelsize,
        labelsize=labelsize,
        legend=False,
        **kwargs)
    twin_ax = _annotate_and_decorate_axis(
        twin_ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        ylabel=right_ylabel,
        wrect=wrect,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        legend=False,
        grid_alpha=0.0,
        **kwargs)
    twin_ax.set_yticklabels(all_algorithm_y, fontsize='large')
    ax.set_yticklabels(all_algorithm_x, fontsize='large')
    twin_ax.set_ylabel(
        right_ylabel,
        fontweight='bold',
        rotation='horizontal',
        fontsize=labelsize)
    ax.set_ylabel(
        left_ylabel,
        fontweight='bold',
        rotation='horizontal',
        fontsize=labelsize)
    twin_ax.set_yticklabels(all_algorithm_y, fontsize=ticklabelsize)
    ax.set_yticklabels(all_algorithm_x, fontsize=ticklabelsize)
    ax.tick_params(axis='both', which='major')
    twin_ax.tick_params(axis='both', which='major')
    ax.spines['left'].set_visible(False)
    twin_ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_coords(-ylabel_x_coordinate, 1.0)
    twin_ax.yaxis.set_label_coords(1 + 0.7 * ylabel_x_coordinate,
                                   1 + 0.6 * ylabel_x_coordinate)

    return fig, ax


MARKERS = ['o', 'v', 's', 'P', 'X', '*', 'H', 'D']


def plot_sample_efficiency_curve(
    frames,
    point_estimates,
    interval_estimates,
    algorithms=None,
    colors=None,
    color_palette=None,
    figsize=(7, 5),
    xlabel=r'Number of Frames (in millions)',
    ylabel='Aggregate Human Normalized Score',
    ax=None,
    labelsize='xx-large',
    ticklabelsize='xx-large',
    **kwargs
):
    """Plots an aggregate metric with CIs as a function of environment frames.
    Args:
      frames: Array or list containing environment frames to mark on the x-axis.
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metric corresponding to the values in `frames`.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Dictionary that maps each algorithm to a color. If None, then this
        mapping is created based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      ax: `matplotlib.axes` object.
      labelsize: Font size of the x-axis label.
      ticklabelsize: Font size of the ticks.
      **kwargs: Arbitrary keyword arguments.
    Returns:
      `axes.Axes` object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))[::-1]
        colors = dict(zip(algorithms, color_palette))
    if kwargs.get('markers', None) is None:
        marker_palette = MARKERS[:len(algorithms)][::-1]
        markers = dict(zip(algorithms, marker_palette))

    for algorithm in algorithms:
        metric_values = point_estimates[algorithm]
        lower, upper = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            marker=markers[algorithm],
            linewidth=kwargs.get('linewidth', 3),
            label=algorithm)
        ax.fill_between(
            frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.1)
    kwargs.pop('linewidth', '2')

    ax = _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs)
    return fig, ax


@logging
def plot_overall_ranks(
    rank_matrix,
    algorithms=None,
    colors=None,
    color_palette=None,
    figsize=(7, 5),
    width=0.985,
    xlabel='Ranking',
    ylabel=None,
    ax=None,
    legend=True,
    labelsize='x-large',
    ticklabelsize='large',
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # num_algo * num_algo
    mean_ranks = np.mean(rank_matrix, axis=0)
    bottom = np.zeros_like(mean_ranks[0])
    labels = list(range(1, len(algorithms) + 1))
    for i, algo in reversed(list(enumerate(algorithms))):
        ax.bar(labels, mean_ranks[i], width, label=algo, bottom=bottom, alpha=0.9)
        bottom += mean_ranks[i] + 0.006

    ax.set_xlabel(xlabel, size=labelsize)
    ax.set_ylabel(ylabel, size=labelsize)

    yticks = np.array(range(0, 101, 20)) * 0.01
    ax.set_xticks(labels)
    ax.set_yticks(yticks)
    ax.set_xticklabels(labels, size=ticklabelsize)
    ax.set_yticklabels(yticks, size=ticklabelsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False,
        left=True, right=False,
        labeltop=False, labelbottom=True, labelleft=True, labelright=False
    )
    if legend:
        ax.legend(
            loc='center right', fancybox=True,
            ncol=1, frameon=True,
            fontsize='x-small',
            bbox_to_anchor=(1.52, 0.47)
        )

    return fig, ax


#  ====================deprecated codes=====================================================
# def plot_each_ranks(save_name, save_dir):
#
#     mean_ranks_all = {}
#     all_ranks_individual = {}
#     for key in ['1m']:
#         exp_score_dict = {algo: read_scores(algo, key, tasks) for algo in algos}
#         exp_score_dict = normalized_scores(exp_score_dict)
#         exp_score_dict = \
#             {alg: convert_to_matrix(scores) for alg, scores in exp_score_dict.items()}
#         all_ranks = get_rank_matrix(exp_score_dict, 200000, algorithms=algos)
#         mean_ranks_all[key] = np.mean(all_ranks, axis=0)
#         all_ranks_individual[key] = all_ranks
#
#     mean_ranks = mean_ranks_all['1m']
#     # @title Plot individual ranks on x tasks
#
#     keys = algos[::-1]
#     labels = list(range(1, len(keys) + 1))
#     width = 1.0       # the width of the bars: can also be len(x) sequence
#
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 10.5))
#     axes = axes.flatten()
#     all_ranks = all_ranks_individual['1m']
#     assert len(tasks) == all_ranks.shape[0]
#     for task in range(len(tasks)):
#         bottom = np.zeros_like(mean_ranks[0])
#         for i, key in enumerate(keys[::-1]):
#             ranks = all_ranks[task]
#             ax = axes[task]
#             ax.bar(labels, ranks[i], width, color=EXP_COLOR_DICT[key],
#                    bottom=bottom, alpha=0.9)
#             bottom += ranks[i]
#             # for label in labels:
#             # perc = int(np.round(mean_ranks[i][label-1] * 100))
#             # ax.text(s= str(perc) + '%', x=label-0.25, y=bottom[label-1] - perc/200,
#             #         color="w", verticalalignment="center",
#             #         horizontalalignment="left", size=10)
#             ax.set_title(tasks[task], fontsize='large')
#
#         if task % 4 == 0:
#             ax.set_ylabel('Fraction (in %)', size='x-large')
#             yticks = np.array(range(0, 101, 20))
#             ax.set_yticks(yticks * 0.01)
#             ax.set_yticklabels(yticks, size='large')
#         else:
#             ax.set_yticklabels([])
#
#         # if task %4 ==  0:
#         #     ax.set_ylabel('Distribution', size='x-large')
#         if task < 8:
#             ax.set_xlabel('')
#         else:
#             ax.set_xlabel('Ranking', size='x-large')
#         ax.set_xticks(labels)
#         # ax.set_ylim(0, 1)
#
#         ax.set_xticklabels(labels, size='large')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         # ax.tick_params(axis='both', which='both', bottom=False, top=False,
#         #                left=False, right=False, labeltop=False,
#         #                labelbottom=True, labelleft=False, labelright=False)
#         left = True
#         ax.tick_params(axis='both', which='both', bottom=False, top=False,
#                        left=left, right=False, labeltop=False,
#                        labelbottom=True, labelleft=left, labelright=False)
#
#     fake_patches = [mpatches.Patch(color=EXP_COLOR_DICT[m], alpha=0.75)
#                     for m in keys]
#     legend = fig.legend(fake_patches, keys, loc='upper center',
#                         fancybox=True, ncol=len(keys), fontsize='x-large')
#     fig.subplots_adjust(wspace=0.1, hspace=0.25)
#     # fig.subplots_adjust(hspace=0.25)
#
#     save_fig(fig, save_name, save_dir)
