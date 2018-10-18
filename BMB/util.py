#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats.mstats as mstats
import scipy.stats as stats
import scipy.linalg as linalg
from collections import namedtuple
import warnings

import pandas as pd

import sqlite3
from sqlite3 import Error
import sys
try:
    import util_plots
except ImportError as e:
    # warnings.warn("Could not import plot utilities:" + str(e))
    pass


def log_progress(sequence, every=None, size=None, name='Items'):
    """https://github.com/alexanderkuk/log-progress

    Arguments:
        sequence {[type]} -- [description]

    Keyword Arguments:
        every {[type]} -- [description] (default: {None})
        size {[type]} -- [description] (default: {None})
        name {str} -- [description] (default: {'Items'})
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def cholesky_invert(A):
    """ Utility method for inverting a symmetric positive definite matrix A using cholesky decomposition.
    
    Arguments:
        A {ndarray} -- Symmetric and positive definite matrix.
    """
    return (linalg.cho_solve(linalg.cho_factor(A), np.identity(A.shape[0], dtype=A.dtype)))
    

def _rank1d(data, keep_na="False"):
    """
    From numpy.stats.mstats.
    Difference: Tie breaking is randomly resolved.

    Arguments:
        data {[type]} -- [description]

    Raises:
        ValueError -- [description]
        NotImplementedError -- [description]

    Returns:
        [type] -- [description]
    """
    data = np.ma.array(data, copy=False)
    n = data.count()
    rk = np.empty(data.size, dtype=float)
    idx = data.argsort()
    rk[idx[:n]] = np.arange(1, n+1)

    repeats = mstats.find_repeats(data.copy())
    for r in repeats[0]:
        condition = (data == r).filled(False)
        rk[condition] = np.random.permutation(rk[condition])
    # keep nas
    if(keep_na):
        rk[np.isnan(data)] = np.nan
    return rk


def cov_from_normscores(Y, scaled=True, fill_na=False):
    """Calculate sample covariance from norm scores.

    Arguments:
        Y {ndarray} -- data

    Keyword Arguments:
        scaled {bool} -- If true, scales the Covariance to [0,1] range (default: {True})
        fill_na {bool} --If true, fills NA normscores with standard normal samples (default: {False})
    """

    Y = pd.DataFrame(Y)
    ranks = Y.rank(axis=0, method='first', na_option='keep')
    N = Y.count()
    U = ranks/(N+1)
    # make it so that draws from ppf return nan for nan entries.
    # the function itself cant handle nans, so just call ppf on values outside of [0,1] instead
    U[np.isnan(U)] = 2
    Z = stats.norm.ppf(U)
    if fill_na:
        Z[np.isnan(Y)] = np.random.normal(size=Y.isna().sum().sum())
    S = Z.T @ Z
    if scaled:
        S = S/S[0, 0]
    return(Z, S)

def calc_MCC(TP, TN, FP, FN):
    return ((TP*TN - FP*FN) / (np.sqrt((TP+FP) * (TP+FN)*(TN+FP)*(TN+FN)))+np.finfo(float).resolution)


def performance_metrics(true_sig, est_sig):
    """Returns metrics for the prediction quality of the matrix as a (named) tuple.

    The tuple keys are:
        - tp
        - fp
        - tn
        - fn
        - precision
        - recall
        - tpr
        - fpr
        - f_score

    Arguments:
        true_sig {np.ndarray} -- True Matrix
        est_sig {np.ndarray} -- Predicted Matrix
    """
    assert(true_sig.shape == est_sig.shape)
    eq = true_sig == est_sig
    neq = true_sig != est_sig
    tp = np.sum(np.logical_and(true_sig, eq))
    tn = np.sum(np.logical_and(np.logical_not(true_sig), eq))
    fp = np.sum(np.logical_and(est_sig, neq))
    fn = np.sum(np.logical_and(np.logical_not(est_sig), neq))

    recall = tp/(tp+fn+np.finfo(float).resolution)
    precision = tp/(tp+fp+np.finfo(float).resolution)
    f_score = 2 / ((1/(recall+np.finfo(float).resolution)) +
                   (1/(precision+np.finfo(float).resolution)))
    sensitivity = recall
    specificity = tn/(tn+fp)
    fpr = fp/(fp+tn)
    performance = namedtuple("perf_metrics", [
                             "tp", "fp", "tn", "fn", "precision", "sensitivity", "specificity", "fpr", "f_score"])

    return(performance(tp, fp, tn, fn, precision, sensitivity, specificity, fpr, f_score))

def performance_metrics2(true_sig, est_sig):
    """Returns metrics for the prediction quality of the matrix as a (named) tuple.

    The tuple keys are:
        - tp
        - fp
        - tn
        - fn
        - precision
        - recall
        - tpr
        - fpr
        - f_score

    Arguments:
        true_sig {np.ndarray} -- True Matrix
        est_sig {np.ndarray} -- Predicted Matrix
    """
    assert(true_sig.shape == est_sig.shape)
    eq = true_sig == est_sig
    neq = true_sig != est_sig
    tp = np.sum((np.abs(true_sig)>0) & (np.sign(true_sig) == np.sign(est_sig)))
    tn = np.sum((np.abs(true_sig)==0) & (np.sign(true_sig) == np.sign(est_sig)))
    fp = np.sum((np.abs(true_sig)==0) & (np.sign(true_sig) != np.sign(est_sig)))
    fn = np.sum((np.abs(true_sig)>0) & (np.sign(true_sig) != np.sign(est_sig)))

    recall = tp/(tp+fn+np.finfo(float).resolution)
    precision = tp/(tp+fp+np.finfo(float).resolution)
    f_score = 2 / ((1/(recall+np.finfo(float).resolution)) +
                   (1/(precision+np.finfo(float).resolution)))
    sensitivity = recall
    specificity = tn/(tn+fp)
    fpr = fp/(fp+tn)
    performance = namedtuple("perf_metrics", [
                             "tp", "fp", "tn", "fn", "precision", "sensitivity", "specificity", "fpr", "f_score"])

    return(performance(tp, fp, tn, fn, precision, sensitivity, specificity, fpr, f_score))



def main():
    raise NotImplementedError(
        "This script is not intended to be run directly.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
