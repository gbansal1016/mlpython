# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 01:21:45 2017

@author: gbans6
"""

"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_val = np.exp(x)
    #print(exp_val)
    denom = np.sum(exp_val, axis=0)
    return np.divide(exp_val, denom)
    #return np.exp(x) / np.sum(np.exp(x),axis=0)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

swmax = softmax(scores)
plt.plot(x, swmax.T, linewidth=2)
plt.plot(x, np.sum(swmax, axis=0),linewidth=2)
plt.legend(['x','1','0.2', 'sum'])
plt.show()
