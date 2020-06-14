#!/usr/bin/evn python
# coding: utf-8
# ========================================================================
# File Name: rank_GroupKFold.py
# Author: weiping
# Created Time: Tue 14 Aug 2018 05:51:29 PM CST
# =======================================================================
'''

'''

from sklearn.model_selection import GroupKFold


class Rank_GroupKFold(GroupKFold):

    def __init__(self, n_splits=3):
        super(Rank_GroupKFold, self).__init__(n_splits)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        groups = X[:, 0]
        for train, test in super(Rank_GroupKFold, self).split(X, y, groups):
            yield train, test
