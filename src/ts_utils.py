"""
ts_utils.py
===========
Reusable time series utility functions for:
SERSA Product Demand Relationship Analysis

These functions are used across notebooks 01–05.
Import from within any notebook in the notebooks/ directory:

    import sys, os
    sys.path.append(os.path.join(os.getcwd(), "..", "src"))
    from ts_utils import build_monthly_pivot, compute_growth_rates, cross_corr_pair, extract_top_pairs

Author : Cesar Enrique Banda Martinez
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Time Series Preparation (notebook 01)
# ---------------------------------------------------------------------------

def build_monthly_pivot(df, date_col="Fecha de Consumo", sku_col="SKU"):
    """
    Aggregate a transaction-level DataFrame into a monthly SKU pivot table.

    Each row in the source DataFrame represents one vending machine dispense
    event. This function counts events per SKU per month and pivots the result
    into a wide-format matrix suitable for correlation analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level DataFrame. Must contain `date_col` (parseable as
        datetime) and `sku_col` (string SKU identifier).
    date_col : str
        Name of the datetime column. Default: "Fecha de Consumo".
    sku_col : str
        Name of the SKU column. Default: "SKU".

    Returns
    -------
    pd.DataFrame
        Wide-format pivot: index = Month (MS frequency), columns = SKUs,
        values = integer transaction counts. Missing SKU-month combinations
        filled with 0.

    Examples
    --------
    >>> pivot = build_monthly_pivot(df)
    >>> pivot.shape
    (50, 92)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    monthly = (
        df.groupby([pd.Grouper(key=date_col, freq="MS"), sku_col])
        .size()
        .reset_index(name="Quantity")
    )

    pivot = (
        monthly
        .pivot(index=date_col, columns=sku_col, values="Quantity")
        .fillna(0)
        .astype(int)
    )

    pivot.index.name = "Month"
    pivot.columns.name = None

    return pivot


def filter_low_activity_skus(pivot, threshold=10):
    """
    Remove SKUs with fewer than `threshold` total transactions from the pivot.

    SKUs with near-zero activity across the full analysis period lack
    sufficient signal for meaningful correlation analysis and are excluded
    before computing any correlation matrices.

    Parameters
    ----------
    pivot : pd.DataFrame
        Monthly pivot table (output of `build_monthly_pivot`).
    threshold : int
        Minimum total transactions required to retain a SKU. Default: 10.

    Returns
    -------
    filtered : pd.DataFrame
        Pivot with low-activity SKUs removed.
    excluded : pd.Series
        SKUs that were removed, with their total transaction counts.

    Examples
    --------
    >>> filtered, excluded = filter_low_activity_skus(pivot, threshold=10)
    >>> print(excluded)
    """
    sku_totals = pivot.sum()
    mask = sku_totals >= threshold
    filtered = pivot.loc[:, mask]
    excluded = sku_totals[~mask].sort_values()
    return filtered, excluded


# ---------------------------------------------------------------------------
# 2. Correlation Analysis (notebooks 02–03)
# ---------------------------------------------------------------------------

def compute_correlation_matrix(pivot, method="pearson"):
    """
    Compute a pairwise correlation matrix across all SKU columns.

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide-format monthly pivot (rows = months, columns = SKUs).
    method : str
        Correlation method passed to pd.DataFrame.corr(). Default: "pearson".

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix of shape (n_skus, n_skus).
    """
    return pivot.corr(method=method)


def compute_growth_rates(pivot):
    """
    Compute month-over-month percentage change for each SKU.

    Removes the common business growth trend so that correlations reflect
    genuine demand co-movement rather than shared upward drift.

    Zero-to-nonzero transitions produce `inf` values (division by zero in
    pct_change). These are replaced with NaN so that `corr()` can handle
    them pairwise without distorting results.

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide-format monthly pivot (rows = months, columns = SKUs).

    Returns
    -------
    pd.DataFrame
        Growth-rate matrix of the same shape. First row is NaN (no prior
        period). Inf values replaced with NaN.

    Examples
    --------
    >>> growth = compute_growth_rates(pivot_filtered)
    >>> growth.shape
    (50, 86)
    """
    growth = pivot.pct_change()
    growth.replace([np.inf, -np.inf], np.nan, inplace=True)
    return growth


def extract_top_pairs(corr_matrix, threshold=0.75, direction="positive"):
    """
    Extract unique SKU pairs above (or below) a correlation threshold.

    Only the upper triangle of the correlation matrix is used to avoid
    duplicate pairs and self-correlations.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square symmetric correlation matrix.
    threshold : float
        Absolute correlation threshold. Default: 0.75.
    direction : str
        "positive" to extract pairs with r >= threshold,
        "negative" to extract pairs with r <= -threshold.
        Default: "positive".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [SKU_A, SKU_B, Pearson_r], sorted by
        Pearson_r descending (positive) or ascending (negative).

    Examples
    --------
    >>> top_pos = extract_top_pairs(corr_matrix, threshold=0.75, direction="positive")
    >>> top_neg = extract_top_pairs(corr_matrix, threshold=0.75, direction="negative")
    """
    upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    unstacked = (
        corr_matrix
        .where(upper_mask)
        .stack()
        .reset_index()
    )
    unstacked.columns = ["SKU_A", "SKU_B", "Pearson_r"]
    unstacked["Pearson_r"] = unstacked["Pearson_r"].round(4)

    if direction == "positive":
        result = (
            unstacked[unstacked["Pearson_r"] >= threshold]
            .sort_values("Pearson_r", ascending=False)
            .reset_index(drop=True)
        )
    elif direction == "negative":
        result = (
            unstacked[unstacked["Pearson_r"] <= -threshold]
            .sort_values("Pearson_r", ascending=True)
            .reset_index(drop=True)
        )
    else:
        raise ValueError("direction must be 'positive' or 'negative'.")

    return result


# ---------------------------------------------------------------------------
# 3. Lead-Lag Analysis (notebook 04)
# ---------------------------------------------------------------------------

def cross_corr_pair(series_a, series_b, max_lag=6):
    """
    Compute Pearson cross-correlation between two time series at integer lags.

    Lags range from -max_lag to +max_lag (inclusive).
    A positive lag k means series_a leads series_b by k periods:
        corr(A_t, B_{t+k})
    A negative lag k means series_b leads series_a by |k| periods.

    Index alignment is handled via reset_index(drop=True) before each
    pairwise concat, so the function works correctly on both RangeIndex
    and DatetimeIndex series.

    Pairs with fewer than 5 valid (non-NaN) overlapping observations
    return NaN for that lag.

    Parameters
    ----------
    series_a : pd.Series
        First time series (e.g. monthly growth rates of SKU A).
    series_b : pd.Series
        Second time series (e.g. monthly growth rates of SKU B).
    max_lag : int
        Maximum lag to compute in both directions. Default: 6.

    Returns
    -------
    dict
        {lag (int): pearson_r (float or NaN)} for lag in [-max_lag, max_lag].

    Examples
    --------
    >>> cc = cross_corr_pair(growth["UVS4040"], growth["UVSVP401"], max_lag=6)
    >>> max(cc, key=lambda k: abs(cc[k] or 0))
    -2
    """
    results = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            a_aligned = series_a.reset_index(drop=True)
            b_aligned = series_b.reset_index(drop=True)
        elif lag > 0:
            # a leads b: align a[t] with b[t+lag]
            a_aligned = series_a.iloc[:-lag].reset_index(drop=True)
            b_aligned = series_b.iloc[lag:].reset_index(drop=True)
        else:
            # b leads a: align b[t] with a[t+|lag|]
            a_aligned = series_a.iloc[-lag:].reset_index(drop=True)
            b_aligned = series_b.iloc[:lag].reset_index(drop=True)

        combined = pd.concat([a_aligned, b_aligned], axis=1).dropna()

        if len(combined) < 5:
            results[lag] = np.nan
        else:
            results[lag] = combined.iloc[:, 0].corr(combined.iloc[:, 1])

    return results


def compute_dominant_lag(cc_dict):
    """
    Return the lag with the highest absolute cross-correlation value.

    Parameters
    ----------
    cc_dict : dict
        Output of `cross_corr_pair`: {lag: pearson_r}.

    Returns
    -------
    tuple
        (dominant_lag, max_abs_r) — the lag and its absolute correlation value.
        Returns (0, NaN) if all values are NaN.

    Examples
    --------
    >>> cc = cross_corr_pair(growth["UVS4040"], growth["UVSVP401"])
    >>> lag, r = compute_dominant_lag(cc)
    >>> print(lag, round(r, 4))
    -2 0.9785
    """
    valid = {k: v for k, v in cc_dict.items() if v is not None and not np.isnan(v)}
    if not valid:
        return 0, np.nan
    dominant = max(valid, key=lambda k: abs(valid[k]))
    return dominant, abs(valid[dominant])