import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR


def discretize_series(
    data: np.ndarray,
    n_bins: int = 8
) -> np.ndarray:
    """
    Discretize a 1D continuous-valued time series into integer bins [0, n_bins-1]
    using uniform binning over the data range.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        Input time series.
    n_bins : int
        Number of bins.

    Returns
    -------
    binned : np.ndarray of ints, shape (n_samples,)
        Discretized series.
    """
    data = np.asarray(data).astype(float)
    if data.ndim != 1:
        raise ValueError("data must be 1D")

    data_min = np.min(data)
    data_max = np.max(data)

    # If constant, everything goes into bin 0
    if data_min == data_max:
        return np.zeros_like(data, dtype=int)

    # Slightly pad the range so max is included
    eps = 1e-12
    edges = np.linspace(data_min - eps, data_max + eps, n_bins + 1)

    # np.digitize with internal bin edges (exclude first & last)
    # returns values in 0..n_bins-1
    binned = np.digitize(data, edges[1:-1])

    return binned.astype(int)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    l: int = 1,
    lag: int = 1,
    n_bins: int = 8,
    base: float = 2.0
) -> float:
    """
    Estimate the transfer entropy TE(source -> target):

        TE_{source -> target} = I( X_{t+lag} ; Y_t^{(l)} | X_t^{(k)} )

    where X is the target series, Y is the source series, using a
    discrete plug-in estimator with uniform binning.

    Parameters
    ----------
    source : array-like, shape (n_samples,)
        Source time series (Y).
    target : array-like, shape (n_samples,)
        Target time series (X).
    k : int
        Embedding length (history) of the target X (X_t, X_{t-1}, ..., X_{t-k+1}).
    l : int
        Embedding length (history) of the source Y (Y_t, Y_{t-1}, ..., Y_{t-l+1}).
    lag : int
        Prediction horizon: X_{t+lag} is the "future" of X.
    n_bins : int
        Number of bins for discretization.
    base : float
        Logarithm base (2 => bits, np.e => nats, etc.)

    Returns
    -------
    te : float
        Estimated transfer entropy TE(source -> target) in the chosen log base.
    """
    x = np.asarray(target).flatten()
    y = np.asarray(source).flatten()

    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have the same length")

    N = len(x)
    if N <= max(k, l) + lag:
        raise ValueError("Time series too short for given k, l, and lag")

    # Discretize both series
    x_d = discretize_series(x, n_bins=n_bins)
    y_d = discretize_series(y, n_bins=n_bins)

    # Count joint occurrences:
    # N_xyz: (x_{t+lag}, X_past, Y_past)
    # N_xy : (X_past, Y_past)
    # N_xx : (x_{t+lag}, X_past)
    # N_x  : (X_past)
    N_xyz: Counter = Counter()
    N_xy: Counter = Counter()
    N_xx: Counter = Counter()
    N_x: Counter = Counter()

    # t is the index of the *current* time step for the past state
    # X_past = (X_t, X_{t-1}, ..., X_{t-k+1})
    # Y_past = (Y_t, Y_{t-1}, ..., Y_{t-l+1})
    # X_next = X_{t+lag}
    start_t = max(k - 1, l - 1)

    for t in range(start_t, N - lag):
        x_next = int(x_d[t + lag])
        x_past = tuple(int(x_d[t - i]) for i in range(k))
        y_past = tuple(int(y_d[t - j]) for j in range(l))

        N_xyz[(x_next, x_past, y_past)] += 1
        N_xy[(x_past, y_past)] += 1
        N_xx[(x_next, x_past)] += 1
        N_x[x_past] += 1

    total = float(sum(N_xyz.values()))
    if total == 0:
        return 0.0

    log_base = np.log(base)
    te = 0.0

    # Plug-in estimator:
    # TE = sum_{x_next, x_past, y_past} p(x_next, x_past, y_past) *
    #          log( [p(x_next | x_past, y_past)] / [p(x_next | x_past)] )
    #     = sum p_xyz * log( p_xyz * p_x / (p_xy * p_xx) )
    for key_xyz, c_xyz in N_xyz.items():
        x_next, x_past, y_past = key_xyz

        c_xy = N_xy[(x_past, y_past)]
        c_xx = N_xx[(x_next, x_past)]
        c_x = N_x[x_past]

        p_xyz = c_xyz / total
        p_xy = c_xy / total
        p_xx = c_xx / total
        p_x = c_x / total

        # Only states actually observed are summed, so no zero probs here
        te += p_xyz * (
            (np.log(p_xyz * p_x) - np.log(p_xy * p_xx)) / log_base
        )

    return float(te)


def pairwise_transfer_entropy(
    data: np.ndarray,
    k: int = 1,
    l: int = 1,
    lag: int = 1,
    n_bins: int = 8,
    base: float = 2.0
) -> np.ndarray:
    """
    Compute pairwise transfer entropy for all pairs of time series.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_series)
        Matrix where each column is one time series.
    k, l, lag, n_bins, base : see `transfer_entropy`.

    Returns
    -------
    TE : np.ndarray, shape (n_series, n_series)
        TE[j, i] = TE(series j -> series i)
        (row = source, column = target)
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be 2D (n_samples, n_series)")

    n_samples, n_series = arr.shape
    TE = np.zeros((n_series, n_series), dtype=float)

    for i in range(n_series):      # target
        for j in range(n_series):  # source
            if i == j:
                continue
            TE[j, i] = transfer_entropy(
                source=arr[:, j],
                target=arr[:, i],
                k=k,
                l=l,
                lag=lag,
                n_bins=n_bins,
                base=base,
            )

    return TE


def plot_te_heatmap(
    TE: np.ndarray,
    labels=None,
    title: str = "Transfer Entropy (source → target)",
    annotate: bool = True,
    fmt: str = ".2f",
    figsize=(6, 5)
):
    """
    Plot a heatmap for a transfer entropy matrix.

    Parameters
    ----------
    TE : np.ndarray, shape (n_series, n_series)
        Transfer entropy matrix where TE[j, i] = TE(series j -> series i)
        (row = source, column = target).
    labels : list of str, optional
        Names of the time series. If None, uses "S0", "S1", ...
    title : str
        Plot title.
    annotate : bool
        If True, write numeric values in each cell.
    fmt : str
        Format string for annotations (e.g. '.2f', '.3f').
    figsize: tuple = (6, 5)

    """
    TE = np.asarray(TE)
    if TE.ndim != 2 or TE.shape[0] != TE.shape[1]:
        raise ValueError("TE must be a square 2D array")

    n = TE.shape[0]
    if labels is None:
        labels = [f"S{i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("Length of labels must match matrix size")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(TE, origin="lower", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Transfer Entropy")

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Axis labels
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title)

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Optional annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                value = TE[i, j]
                ax.text(
                    j, i,
                    format(value, fmt),
                    ha="center", va="center",
                    fontsize=8
                )

    plt.tight_layout()
    plt.show()
    


