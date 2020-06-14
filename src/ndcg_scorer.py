#!/usr/bin/evn python
# coding: utf-8
# ========================================================================
# File Name: ndcg_scorer.py
# Author: weiping
# Created Time: Tue 14 Aug 2018 05:16:07 PM CST
# =======================================================================
'''

'''
import numpy as np
import pandas as pd


def dcg_at_k(r, k, method=0):
    """
    Basic implementation of the DCG metric,
    credits to: https://gist.github.com/bwhite/3726239

    Parameters
    ----------
    r : (1d-array) Relavances, sorted as in the result (i.e. to predicted rank)

    k : (int) evaluate up till this position

    method : (int, 0 or 1) Weights to use when calculating relevance

    Returns
    -------
    dcg (float) : the calculated dcg value
    """

    # Pad r if necessary
    if r.shape[0] < k:
        padding = k - r.shape[0]
        r = np.pad(r, (0, padding), 'constant', constant_values=0)

    # Only consider k elements
    r = r[:k]
    if r.size:
        if method == 0:
            return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Basic implementation of the nDCG metric,
    credits to: https://gist.github.com/bwhite/3726239

    Parameters
    ----------
    r : (1d-array) Relavances, sorted as in the result (i.e. to predicted rank)

    k : (int) evaluate up till this position

    method : (int, 0 or 1) Weights to use when calculating relevance

    Returns
    -------
    ndcg (float) : the calculated ndcg value
    """
    dcg_max = dcg_at_k(np.sort(r)[::-1], k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def _ndcg_at_k_scoring(est, X, y, at):
    group = X[:, 0]
    predict_rank = est.predict(X)

    def query_ndcg(query_df):
        r = query_df.sort_values(by='p', ascending=False)['y'].values
        return ndcg_at_k(r, at)
    gpy = pd.DataFrame({'g': group, 'p': predict_rank, 'y': y})
    query_ndcgs = gpy.groupby('g').apply(query_ndcg)
    return np.mean(query_ndcgs)


def ndcg_at_3_scoring(est, X, y):
    return _ndcg_at_k_scoring(est, X, y, 3)


def ndcg_at_5_scoring(est, X, y):
    return _ndcg_at_k_scoring(est, X, y, 5)


def ndcg_at_10_scoring(est, X, y):
    return _ndcg_at_k_scoring(est, X, y, 10)
