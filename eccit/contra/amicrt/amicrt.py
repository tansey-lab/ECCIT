"""
This code is primarily adapted from CONTRA:
https://github.com/rajesh-lab/contra-public
with modifications to support continuous feature spaces.
"""

import os
import warnings
from itertools import product, repeat

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer

from .conditionals import CompleteConditional, ContinuousConditional
from .utils import binaryLoss, ltgteq, monteCarloEntropy, continuousComboMSE
from .statistics import CorrelationStatistic, ModelBasedRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def smartComboStat(x, y, m1, m2):
    """Automatically choose between binary and continuous statistics based on y."""
    if len(np.unique(y)) <= 2:
        # Binary response - use original comboProb
        return comboProb(x, y, m1, m2)
    else:
        # Continuous response - use MSE-based statistic
        return continuousComboMSE(x, y, m1, m2)


def initializeHelper(inp):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    testStat, x, y = inp
    return testStat.fit(x, y)


def fitNullHelper(inp):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    (testStat, j, Q_j), x, y = inp

    xTilde = x.copy()
    xTilde[:, j] = Q_j.sample(np.concatenate([x[:, :j], x[:, j + 1:]], axis=1))

    return (j, testStat.fit(xTilde, y))


def fitEvaluateHelper(inp):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    (newStat, oldStat, j, Q_j), xTr, yTr, xTe, yTe, refitStatistic = inp

    xTildeTr = xTr.copy()
    xTildeTr[:, j] = Q_j.sample(
        np.concatenate([xTr[:, :j], xTr[:, j + 1:]], axis=1))
    xTildeTe = xTe.copy()
    xTildeTe[:, j] = Q_j.sample(
        np.concatenate([xTe[:, :j], xTe[:, j + 1:]], axis=1))

    if refitStatistic:
        nullStat = newStat.fit(xTildeTr, yTr)
        nullStatScore = nullStat.score(xTildeTe, yTe)
    else:
        nullStatScore = oldStat.score(xTildeTe, yTe)

    return (j, nullStatScore)


class CRT(object):
    def __init__(self,
                 CompleteConditionalClass=CompleteConditional,
                 CompleteConditionalArgs=dict(),
                 TestStatisticClass=CorrelationStatistic,
                 TestStatisticArgs=dict(),
                 ordering='>',
                 M=30,
                 tqdm=lambda x: x,
                 refitStatistic=True,
                 pool=None, chunksize=2, conservative=False):

        self.CompleteConditionalClass = CompleteConditionalClass
        self.CompleteConditionalArgs = CompleteConditionalArgs

        self.TestStatisticClass = TestStatisticClass
        self.TestStatisticArgs = TestStatisticArgs

        self.M = M
        self.tqdm = tqdm
        self.refitStatistic = refitStatistic

        self.ordering = ordering
        self.pool = pool
        self.chunksize = chunksize

        self.initialized = False
        self.conservative = conservative

    def initialize(self, x, y, is_discrete=None, test_inds=None):
        """
        is_discrete : dictionary of discrete indicators
        """
        _, d = x.shape

        # check if only a subset of features need to be tested
        if test_inds is None:
            test_inds = np.arange(d)
        else:
            test_inds = np.array(test_inds)
        self.test_inds = test_inds

        # store discrete feature indicators
        if is_discrete is not None:
            self.is_discrete = {j: is_discrete[j] for j in self.test_inds}
        else:
            self.is_discrete = {j: False for j in self.test_inds}

        # initialize and fit complete conditional models
        # initialize
        self.Q = {
            j: self.CompleteConditionalClass(**self.CompleteConditionalArgs,
                                             is_discrete=self.is_discrete[j])
            for j in self.test_inds
        }
        # fit
        if self.pool is None:
            for j in self.tqdm(self.test_inds,
                               desc='Fitting q(x_j | x_{-j})',
                               leave=False):
                self.Q[j].fit(np.concatenate([x[:, :j], x[:, j + 1:]], axis=1),
                              x[:, j])
        else:
            # build list of inputs for multiprocessing map
            inputs = [(self.Q[j],
                       np.concatenate([x[:, :j], x[:, j + 1:]], axis=1), x[:,
                                                                           j])
                      for j in self.test_inds]
            # get list of outputs
            chunksize = min(
                1 + int(len(inputs) / self.pool._processes), self.chunksize)
            res = list(
                self.tqdm(self.pool.imap(initializeHelper, inputs, chunksize=chunksize),
                          total=len(self.test_inds),
                          desc='Fitting q(x_j | x_{-j})',
                          leave=False))
            # store complete conditional models
            for j, Q_j in zip(self.test_inds, res):
                self.Q[j] = Q_j

        # initialize test statistic for each feature
        self.testStats = {
            j: self.TestStatisticClass(**self.TestStatisticArgs, j=j)
            for j in self.test_inds
        }

        if self.pool is None:
            for j in self.tqdm(self.test_inds,
                               desc='Computing test statistics',
                               leave=False):
                self.testStats[j].fit(x, y)
        else:
            n_test_inds = len(self.test_inds)
            inputs = zip([self.testStats[j] for j in self.test_inds],
                         repeat(x, n_test_inds), repeat(y, n_test_inds))
            chunksize = min(
                1 + int(n_test_inds / self.pool._processes), self.chunksize)
            res = list(
                self.tqdm(self.pool.imap(initializeHelper, inputs, chunksize=chunksize),
                          total=n_test_inds,
                          desc='Computing test statistics',
                          leave=False))

            for j, t_j in zip(self.test_inds, res):
                self.testStats[j] = t_j

        self.initialized = True

    def fit_evaluate(self, xTr, yTr, xTe, yTe):
        assert self.initialized, 'Test is not initialized. Please call .initialize() first'
        """
        low memory version of .fitNullModels() and .evaluate()
        """
        nTr, d = xTr.shape
        nTe, _ = xTe.shape

        # ground truth test statistic
        self.tStars = {
            j: self.testStats[j].score(xTe, yTe)
            for j in self.test_inds
        }

        self.nullT = {j: [] for j in self.test_inds}

        if self.pool is None:
            for j in self.tqdm(
                    self.test_inds,
                    desc='Fitting null model for q(y | ~x_j , x_{-j})',
                    leave=False):
                for m in range(self.M):
                    xTildeTr = xTr.copy()
                    xTildeTr[:, j] = self.Q[j].sample(
                        np.concatenate([xTr[:, :j], xTr[:, j + 1:]], axis=1))

                    xTildeTe = xTe.copy()
                    xTildeTe[:, j] = self.Q[j].sample(
                        np.concatenate([xTe[:, :j], xTe[:, j + 1:]], axis=1))

                    if self.refitStatistic:
                        # CRT
                        # fit nullModels
                        nullStat = self.TestStatisticClass(
                            **self.TestStatisticArgs, j=j).fit(xTildeTr, yTr)
                        self.nullT[j].append(nullStat.score(xTildeTe, yTe))
                    else:
                        # HRT
                        self.nullT[j].append(self.testStats[j].score(
                            xTildeTe, yTe))
        else:
            # number of features to test
            n_test_inds = len(self.test_inds)
            # number of total model experiments
            dM = n_test_inds * self.M

            input1 = [(self.TestStatisticClass(**self.TestStatisticArgs, j=j),
                       self.testStats[j], j, self.Q[j]) for j in self.test_inds
                      for m in range(self.M)]
            inputs = zip(input1, repeat(xTr, dM), repeat(yTr, dM),
                         repeat(xTe, dM), repeat(yTe, dM),
                         repeat(self.refitStatistic, dM))

            chunksize = min(1 + int(dM / self.pool._processes), self.chunksize)
            outputs = list(
                self.tqdm(self.pool.imap(fitEvaluateHelper, inputs, chunksize=chunksize),
                          total=dM,
                          desc='Fitting null model for q(y | ~x_j , x_{-j})',
                          leave=False))
            for j, nullStatScore in outputs:
                self.nullT[j].append(nullStatScore)

        # return p-values
        self.pvalues = {j: 0.0 for j in self.test_inds}
        for j in self.tqdm(self.test_inds,
                           desc='Computing p-values',
                           leave=False):
            numReps = len(self.nullT[j])
            for i in range(numReps):
                # For ordering='<' (lower is better), count how many null values are <= observed
                # For ordering='>' (higher is better), count how many null values are >= observed
                if self.ordering == '<':
                    self.pvalues[j] += 1 if self.nullT[j][i] <= self.tStars[j] else 0
                else:
                    self.pvalues[j] += 1 if ltgteq(
                        self.tStars[j], self.nullT[j][i], self.ordering) else 0

            if self.conservative:
                self.pvalues[j] = (self.pvalues[j] + 1) / (numReps + 1)
            else:
                self.pvalues[j] /= numReps
        return self.pvalues

    def __repr__(self):
        method = 'CRT' if self.refitStatistic else 'HRT'
        return f'{method}({self.CompleteConditionalClass, self.TestStatisticClass})'


class FastCRT(object):
    def __init__(self,
                 CompleteConditionalClass=CompleteConditional,
                 CompleteConditionalArgs=dict(),
                 TestStatisticClass=CorrelationStatistic,
                 TestStatisticArgs=dict(),
                 ordering='>',
                 M=30,
                 tqdm=lambda x: x,
                 refitStatistic=True,
                 pool=None, chunksize=2, conservative=False):

        self.CompleteConditionalClass = CompleteConditionalClass
        self.CompleteConditionalArgs = CompleteConditionalArgs

        self.TestStatisticClass = TestStatisticClass
        self.TestStatisticArgs = TestStatisticArgs

        self.M = M
        self.tqdm = tqdm
        self.refitStatistic = refitStatistic

        self.ordering = ordering
        self.pool = pool
        self.chunksize = chunksize

        self.initialized = False
        self.conservative = conservative

    def initialize(self, x, y, is_discrete=None, test_inds=None):
        """
        is_discrete : dictionary of discrete indicators
        """
        _, d = x.shape

        # check if only a subset of features need to be tested
        if test_inds is None:
            test_inds = np.arange(d)
        else:
            test_inds = np.array(test_inds)
        self.test_inds = test_inds

        # store discrete feature indicators
        if is_discrete is not None:
            self.is_discrete = {j: is_discrete[j] for j in self.test_inds}
        else:
            self.is_discrete = {j: False for j in self.test_inds}

        # initialize and fit complete conditional models
        # initialize
        self.Q = {
            j: self.CompleteConditionalClass(**self.CompleteConditionalArgs,
                                             is_discrete=self.is_discrete[j])
            for j in self.test_inds
        }
        # fit
        if self.pool is None:
            for j in self.tqdm(self.test_inds,
                               desc='Fitting q(x_j | x_{-j})',
                               leave=False):
                self.Q[j].fit(np.concatenate([x[:, :j], x[:, j + 1:]], axis=1),
                              x[:, j])
        else:
            # build list of inputs for multiprocessing map
            inputs = [(self.Q[j],
                       np.concatenate([x[:, :j], x[:, j + 1:]], axis=1), x[:,
                                                                           j])
                      for j in self.test_inds]
            # get list of outputs
            chunksize = min(
                1 + int(len(inputs) / self.pool._processes), self.chunksize)
            res = list(
                self.tqdm(self.pool.imap(initializeHelper, inputs, chunksize=chunksize),
                          total=len(self.test_inds),
                          desc='Fitting q(x_j | x_{-j})',
                          leave=False))
            # store complete conditional models
            for j, Q_j in zip(self.test_inds, res):
                self.Q[j] = Q_j

        # initialize test statistic for each feature
        self.testStats = {
            j: self.TestStatisticClass(**self.TestStatisticArgs, j=j)
            for j in self.test_inds
        }

        if self.pool is None:
            for j in self.tqdm(self.test_inds,
                               desc='Computing test statistics',
                               leave=False):
                self.testStats[j].fit(x, y)
        else:
            n_test_inds = len(self.test_inds)
            inputs = zip([self.testStats[j] for j in self.test_inds],
                         repeat(x, n_test_inds), repeat(y, n_test_inds))
            chunksize = min(
                1 + int(n_test_inds / self.pool._processes), self.chunksize)
            res = list(
                self.tqdm(self.pool.imap(initializeHelper, inputs, chunksize=chunksize),
                          total=n_test_inds,
                          desc='Computing test statistics',
                          leave=False))

            for j, t_j in zip(self.test_inds, res):
                self.testStats[j] = t_j

        self.initialized = True

    def fitNullModels(self, x, y):
        assert self.initialized, 'Test is not initialized. Please call .initialize() first'

        # initialize and fit nullModels
        # if HRT, don't fit null models
        if not self.refitStatistic:
            return

        self.nullModels = {j: [] for j in self.test_inds}

        if self.pool is None:
            for j in self.tqdm(
                    self.test_inds,
                    desc='Fitting null model for q(y | ~x_j , x_{-j})',
                    leave=False):
                # fit only 1 null model
                for _ in range(1):
                    # sample Q_j(x_j | x_{-j})
                    xTilde = x.copy()
                    xTilde[:, j] = self.Q[j].sample(
                        np.concatenate([x[:, :j], x[:, j + 1:]], axis=1))

                    # fit nullModels
                    self.nullModels[j].append(
                        self.TestStatisticClass(**self.TestStatisticArgs,
                                                j=j).fit(xTilde, y))
        else:
            # fit only 1 null model
            n_test_inds = len(self.test_inds)
            dM = n_test_inds * 1
            input1 = [(self.TestStatisticClass(**self.TestStatisticArgs,
                                               j=j), j, self.Q[j])
                      for j in self.test_inds for m in range(1)]
            inputs = zip(input1, repeat(x, dM), repeat(y, dM))
            chunksize = min(1 + int(dM / self.pool._processes), self.chunksize)
            outputs = list(
                self.tqdm(self.pool.imap(fitNullHelper, inputs, chunksize=chunksize),
                          total=dM,
                          desc='Fitting null model for q(y | ~x_j , x_{-j})',
                          leave=False))
            for j, testStat in outputs:
                self.nullModels[j].append(testStat)

        # flatten list of null models
        tmp = {j: self.nullModels[j][0] for j in self.test_inds}
        self.nullModels = tmp

    def fit_evaluate(self, xTr, yTr, xTe, yTe):
        assert self.initialized, 'Test is not initialized. Please call .initialize() first'
        """
        Evaluate models for FAST-AMI-CRT. Returns p-values
        """
        # fit null models
        self.fitNullModels(xTr, yTr)

        _, d = xTe.shape

        # initialize hypothesis test values
        self.tStars = {j: 0 for j in self.test_inds}
        self.nullT = {j: [] for j in self.test_inds}

        if self.pool is None:
            for j in self.tqdm(
                    self.test_inds,
                    desc='Computing test statistics and null distributions',
                    leave=False):
                # compute ground truth test statistic
                self.tStars[j] = smartComboStat(xTe, yTe, self.testStats[j].model,
                                                self.nullModels[j].model)

                # compute null statistics
                for _ in range(self.M):
                    # get a null dataset
                    xTildeTe = xTe.copy()
                    xTildeTe[:, j] = self.Q[j].sample(
                        np.concatenate([xTe[:, :j], xTe[:, j + 1:]], axis=1))

                    null_stat = smartComboStat(xTildeTe, yTe,
                                               self.testStats[j].model,
                                               self.nullModels[j].model)
                    self.nullT[j].append(null_stat)
        else:
            # compute ground truth test statistics
            # keep order based on self.test_inds
            n_test_inds = len(self.test_inds)
            input1 = [self.testStats[j] for j in self.test_inds]
            input2 = [self.nullModels[j] for j in self.test_inds]
            inputs = zip(repeat(xTe, n_test_inds), repeat(yTe, n_test_inds),
                         input1, input2)

            chunksize = min(
                1 + int(n_test_inds / self.pool._processes), self.chunksize)
            res = list(
                self.tqdm(self.pool.imap(test_stat_helper, inputs, chunksize=chunksize),
                          total=d,
                          desc='Computing test statistics',
                          leave=False))

            for j, t_j in zip(self.test_inds, res):
                self.tStars[j] = t_j

            # compute null statistics
            dM = n_test_inds * self.M

            # run parallel null computation
            inputs = zip(
                repeat(xTe, dM),
                repeat(yTe, dM),
                product([(self.Q[j], self.testStats[j], self.nullModels[j], j)
                         for j in self.test_inds], range(self.M)),
            )
            chunksize = min(1 + int(dM / self.pool._processes), self.chunksize)
            results = list(
                self.tqdm(self.pool.imap(fast_null_helper, inputs, chunksize=chunksize),
                          total=dM,
                          desc='Computing null distributions',
                          leave=False))

            # gather results of null computation
            ctr = 0
            for j in self.test_inds:
                for m in range(self.M):
                    self.nullT[j].append(results[ctr])
                    ctr += 1

        # return p-values
        self.pvalues = {j: 0.0 for j in self.test_inds}
        for j in self.tqdm(self.test_inds,
                           desc='Computing p-values',
                           leave=False):
            numReps = len(self.nullT[j])
            for i in range(numReps):
                # For ordering='<' (lower is better), count how many null values are <= observed
                # For ordering='>' (higher is better), count how many null values are >= observed
                if self.ordering == '<':
                    self.pvalues[j] += 1 if self.nullT[j][i] <= self.tStars[j] else 0
                else:
                    self.pvalues[j] += 1 if ltgteq(
                        self.tStars[j], self.nullT[j][i], self.ordering) else 0
            if self.conservative:
                self.pvalues[j] = (self.pvalues[j] + 1) / (numReps + 1)
            else:
                self.pvalues[j] /= numReps
        return self.pvalues

    def __repr__(self):
        method = 'FAST-CRT'
        return f'{method}({self.CompleteConditionalClass, self.TestStatisticClass})'


def fast_null_helper(inp):
    xTe, yTe, ((Q_j, ts_j, nm_j, j), m) = inp
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # get a null dataset
    xTildeTe = xTe.copy()
    xTildeTe[:, j] = Q_j.sample(
        np.concatenate([xTe[:, :j], xTe[:, j + 1:]], axis=1))

    null_stat = smartComboStat(xTildeTe, yTe, ts_j.model, nm_j.model)
    return null_stat


def test_stat_helper(inp):
    x, y, m1, m2 = inp

    return smartComboStat(x, y, m1.model, m2.model)


def safelog(x, eps=1e-4):
    out = np.zeros_like(x)
    a = x != 0
    out[a] = np.log(x[a])
    out[~a] = np.log(eps)
    return out


def comboProb(x, y, m1, m2, log_fn=safelog):
    assert len(np.unique(y)) == 2, 'y must be binary'
    out = 0
    tmp = 1 / 2 * m1.predict_proba(x) + 1 / 2 * m2.predict_proba(x)
    for i in range(len(x)):
        out += log_fn(tmp[i, y[i]])
    return -1 * out / len(x)
