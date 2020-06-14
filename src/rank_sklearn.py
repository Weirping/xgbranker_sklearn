#!/usr/bin/evn python
# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, E0012, R0912
# ========================================================================
# File Name: rank_sklearn.py
# Author: weiping
# mail: zhangweiping1988@gmail.com
# Created Time: Fri 03 Aug 2018 06:23:54 PM CST
# =======================================================================
'''
Scikit-Learn Wrapper interface for XGBoost rank.
'''

import numpy as np
from xgboost import DMatrix, train
from xgboost import XGBModel

from ndcg_scorer import ndcg_at_3_scoring
from rank_util import _preprare_data_in_groups


class RankerMixin(object):
    """Mixin class for all ranking in scikit-learn."""
    _estimator_type = "ranker"

    def score(self, X, y, sample_weight=None, at=3):
        """Returns the ndcg@at on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features + 1)
          Test samples. Feature matrix with the first column the group label

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
          True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
          Sample weights.

        at : int, default = 3
                ndcg@at

        Returns
        -------
        score : float
                ndcg@at
        """
        return ndcg_at_3_scoring(self, X, y)


class XGBRanker(XGBModel, RankerMixin):
    """Implementation of sklearn API for XGBoost Ranking

        Parameters
    ----------
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    n_jobs : int
        Number of parallel threads used to run xgboost.  (replaces nthread)
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each split, in each level.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    scale_pos_weight : float
        Balancing of positive and negative weights.
    base_score:
        The initial prediction score of all instances, global bias.
    random_state : int
        Random number seed.  (replaces seed)
    missing : float, optional
        Value in the data which needs to be present as a missing value. If
        None, defaults to np.nan.
    **kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters can
        be found here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md.
        Attempting to set a parameter via the constructor args and **kwargs dict simultaneously
        will result in a TypeError.
        Note:
            **kwargs is unsupported by Sklearn.  We do not guarantee that parameters passed via
            this argument will interact properly with Sklearn.

    Note
    ----
    A custom objective function can be provided for the ``objective``
    parameter. In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``:

    y_true: array_like of shape [n_samples]
        The target values
    y_pred: array_like of shape [n_samples]
        The predicted values

    grad: array_like of shape [n_samples]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples]
        The value of the second derivative for each sample point
    """

    def __init__(self,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 silent=True,
                 n_jobs=-1,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 random_state=0,
                 missing=None,
                 **kwargs):

        super(XGBRanker, self).__init__(
            max_depth, learning_rate, n_estimators, silent, 'rank:pairwise', 'gbtree', n_jobs, None,
            gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel,
            reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state, None, missing)

    def get_xgb_params(self):
        """
        Get xgboost type parameters.
        xgb_params is based on the signature of __init__.
        in our case, "objective" absent in the signature of __init__
        so we need to add it.
        """
        xgb_params = self.get_params()

        xgb_params['silent'] = 1 if self.silent else 0

        if self.nthread <= 0:
            xgb_params.pop('nthread', None)
            # add default params for ranking
        xgb_params["objective"] = self.objective
        xgb_params["booster"] = self.booster
        return xgb_params

    def fit(self,
            X,
            y,
            sample_weight=None,
            eval_set=None,
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=True,
            xgb_model=None):
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix with the first feature containing a group indicator
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored xgb model or 'Booster' instance Xgb model to be
            loaded before training (allows training continuation).
        """

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        group_sizes, _, X_features, y, sample_weight = _preprare_data_in_groups(X, y, sample_weight)
        params = self.get_xgb_params()
        evals_result = {}
        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        if sample_weight is not None:
            train_dmatrix = DMatrix(X_features, label=y, weight=sample_weight, missing=self.missing)
        else:
            train_dmatrix = DMatrix(X_features, label=y, missing=self.missing)
        train_dmatrix.set_group(group_sizes)

        self._Booster = train(
            params,
            train_dmatrix,
            self.n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=None,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=xgb_model)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    def predict(self, X, output_margin=False, ntree_limit=0):
        '''
        X (array_like) – The dmatrix storing the input.
        output_margin (bool) – Whether to output the raw untransformed margin value.
        ntree_limit (int) – Limit number of trees in the prediction; defaults to 0 (use all trees).
        '''
        sizes, group_indices, X_features, _, _ = _preprare_data_in_groups(X)
        test_dmatrix = DMatrix(X_features, missing=self.missing)
        test_dmatrix.set_group(sizes)
        rank_values = self.get_booster().predict(
            test_dmatrix, output_margin=output_margin, ntree_limit=ntree_limit)
        revert_group_indices = np.arange(len(group_indices))[group_indices.argsort()]
        rank_values = rank_values[revert_group_indices]
        return rank_values

    def apply(self, X, ntree_limit=0):
        """Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
            For each datapoint x in X and for each tree, return the index of the
            leaf x ends up in. Leaves are numbered within
            ``[0; 2**(self.max_depth+1))``, possibly with gaps in the numbering.
        """
        sizes, group_indices, X_features, _, _ = _preprare_data_in_groups(X)
        test_dmatrix = DMatrix(X_features, missing=self.missing)
        test_dmatrix.set_group(sizes)
        X_leaves = self.get_booster().predict(test_dmatrix, pred_leaf=True, ntree_limit=ntree_limit)
        revert_group_indices = np.arange(len(group_indices))[group_indices.argsort()]
        X_leaves = X_leaves[revert_group_indices, :]
        return X_leaves
