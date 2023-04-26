# limitations under the License.
"""Aggregate Performance Estimators."""

# yanked and modified from https://github.com/google-research/rliable/blob/master/rliable/metrics.py

import numpy as np
import scipy.stats


def aggregate_mean(scores: np.ndarray):
    """Computes mean of sample mean scores per task.
    Args:
      scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
        represent the score on run `n` of task `m`.
    Returns:
      Mean of sample means.
    """
    if scores.ndim == 1:
        scores = scores[:, None]
    mean_task_scores = np.mean(scores, axis=0, keepdims=False)
    return np.mean(mean_task_scores, axis=0)


def aggregate_median(scores: np.ndarray):
    """Computes median of sample mean scores per task.
    Args:
      scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
        represent the score on run `n` of task `m`.
    Returns:
      Median of sample means.
    """
    mean_task_scores = np.mean(scores, axis=0, keepdims=False)
    return np.median(mean_task_scores, axis=0)


def aggregate_optimality_gap(scores: np.ndarray, gamma=1):
    """Computes optimality gap across all runs and tasks.
    Args:
      scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
        represent the score on run `n` of task `m`.
      gamma: Threshold for optimality gap. All scores above `gamma` are clipped
       to `gamma`.
    Returns:
      Optimality gap at threshold `gamma`.
    """
    return gamma - np.mean(np.minimum(scores, gamma))


def aggregate_iqm(scores: np.ndarray):
    """Computes the interquartile mean across runs and tasks.
    Args:
      scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
        represent the score on run `n` of task `m`.
    Returns:
      IQM (25% trimmed mean) of scores.
    """
    return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None)


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray):
    """Overall Probability of imporvement of algorithm `X` over `Y`.
    Args:
      scores_x: A matrix of size (`num_runs_x` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `X`.
      scores_y: A matrix of size (`num_runs_y` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `Y`.
    Returns:
        P(X_m > Y_m) averaged across tasks.
    """
    num_tasks = scores_x.shape[1]
    task_improvement_probabilities = []
    num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
    for task in range(num_tasks):
        if np.array_equal(scores_x[:, task], scores_y[:, task]):
            task_improvement_prob = 0.5
        else:
            task_improvement_prob, _ = scipy.stats.mannwhitneyu(
                scores_x[:, task], scores_y[:, task], alternative='greater')
            task_improvement_prob /= (num_runs_x * num_runs_y)
        task_improvement_probabilities.append(task_improvement_prob)
    return np.mean(task_improvement_probabilities)
