"""
This code is primarily adapted from CONTRA:
https://github.com/rajesh-lab/contra-public
with modifications to support continuous feature spaces.
"""

import numpy as np

def monteCarloEntropy(model, x, y):
    out = 0
    yHatProbs = model.predict_proba(x)

    numToRemove = 0
    for i in range(len(y)):
        p_y_x = yHatProbs[i, y[i]]
        if p_y_x == 0:
            numToRemove += 1
        else:
            out -= np.log(p_y_x)

    # warning: this can result in error if numToRemove = len(y)
    return out / (len(y) - numToRemove)


def binaryLoss(model, x, y):
    """
    assumes model has a .predict() method
    """
    return (model.predict(x) == y).mean()


def continuousComboMSE(x, y, m1, m2):
    """Continuous version of comboProb using MSE for regression models."""
    # Combine predictions from two regression models
    pred1 = m1.predict(x)
    pred2 = m2.predict(x)
    combined_pred = 0.5 * pred1 + 0.5 * pred2

    # Compute mean squared error
    mse = np.mean((combined_pred - y) ** 2)
    return mse


def ltgteq(a, b, operation):
    if operation == '>':
        return a > b
    if operation == '<':
        return a < b
    if operation == '==':
        return a == b
    if operation == '>=':
        return a >= b
    if operation == '<=':
        return a <= b
