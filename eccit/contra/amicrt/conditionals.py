"""
This code is primarily adapted from CONTRA:
https://github.com/rajesh-lab/contra-public
with modifications to support continuous feature spaces.
"""

import numpy as np
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import KBinsDiscretizer

class CompleteConditional(object):
    def __init__(self,
                 is_discrete=False,
                 n_bins=20,
                 modelClass=MLPClassifier,
                 modelArgs=dict(hidden_layer_sizes=(4, 4),
                                activation='relu',
                                max_iter=50),
                 name='nnCCK'):
        """
        modelClass must contain the following methods:
            - fit
            - predict_proba
        """
        self.name = name

        # if x_j is discrete
        self.is_discrete = is_discrete
        # number of bins to discretize x_j into
        self.n_bins = n_bins

        self.modelClass = modelClass
        self.modelArgs = modelArgs

        self.isFit = False

    def fit(self, X_mj, x_j):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        if not self.is_discrete:
            # discretize
            est = KBinsDiscretizer(n_bins=self.n_bins,
                                   encode='ordinal',
                                   strategy='uniform')
            # save discretization model to do inverse transform later
            self.est = est
            x_j = est.fit_transform(x_j.reshape(-1, 1)).flatten()

        self.clf = self.modelClass(**self.modelArgs)
        self.clf.fit(X_mj, x_j)
        self.isFit = True

        return self

    def sample(self, X_mj):
        assert self.isFit, 'model must be fit before calling .sample() method'
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        preds = self.clf.predict_proba(X_mj)
        if not self.is_discrete:
            out = []
            for row in preds:
                out.append(np.where(np.random.multinomial(1, row))[0][0])
            out = np.array(out).reshape(-1, 1)
            return self.est.inverse_transform(out).flatten()
        else:
            x_j_tilde = np.random.binomial(1, preds[:, 1])
            return x_j_tilde

    def __repr__(self):
        return f"{self.name}[{'Discrete' if self.is_discrete else 'Discretized'}]"


class ContinuousConditional(object):
    def __init__(self,
                 modelClass=MLPRegressor,
                 modelArgs=dict(hidden_layer_sizes=(4, 4),
                                activation='relu',
                                max_iter=50),
                 name='nnCCR',
                 is_discrete=False):
        """Continuous conditional sampler using MLP regression."""
        self.name = name
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.is_discrete = is_discrete  # Store but ignore for continuous conditionals
        self.isFit = False

    def fit(self, X_mj, x_j):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        self.reg = self.modelClass(**self.modelArgs)
        self.reg.fit(X_mj, x_j)
        preds = self.reg.predict(X_mj)
        self.residuals = x_j - preds
        self.isFit = True
        return self

    def sample(self, X_mj):
        assert self.isFit, 'model must be fit before calling .sample() method'
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        preds = self.reg.predict(X_mj)
        if self.residuals.size:
            resampled = np.random.choice(self.residuals, size=X_mj.shape[0], replace=True)
        else:
            resampled = np.random.normal(size=X_mj.shape[0])
        return preds + resampled

    def __repr__(self):
        return f"{self.name}[Continuous]"
