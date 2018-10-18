#!/usr/bin/env python3
# coding: utf-8

# scipy stack
import numpy as np
import scipy.stats.mstats as mstats
import scipy.stats as stats


import seaborn as sns
# for loading data
import json
import os

# misc
from collections import namedtuple

import logging
import sys

# plots
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf

import networkx as nx

from bokeh.layouts import gridplot
from bokeh.io import show, output_file


import warnings

dir_path = os.path.dirname(__file__)

from BMB import MB
from BMB import util_plots

#################################


def compare_Gibbs_SA(SEED, NSCAN, BURNIN, lambda_, lambda_SA, thresh_low, size=300, PLUGIN_THRESHOLD=40):
    graph_plots = []
    for file_no in range(1, 4):
        print("##########################")
        print("File {0}".format(file_no))
        np.random.seed(SEED)

        # load data
        num_str = "0"+str(file_no) if file_no < 10 else str(file_no)
        file_name = "data"+num_str+".json"
        file_path = "example_data/"+file_name
        with open(os.path.join(dir_path, file_path)) as json_data:
            data_json = json.load(json_data)
        logging.debug(file_path)

        data = np.array(data_json["data"])
        inv_cov = np.array(data_json["inv.cov"])
        p = data_json["p"][0]
        q = data_json["q"][0]
        # S = np.array(data_json["S"])
        # n = np.shape(data)[0]

        # Gibbs MB
        MB_gibbs = MB.BMB_gibbs(
            X=data, p=p, q=q, lambda_=lambda_,
            burnin=BURNIN,  NSCAN=NSCAN,
            DO_COPULA=True,
            TRUE_INVCOV=inv_cov,
            plugin_threshold=PLUGIN_THRESHOLD)
        MB_gibbs.run()
        adj_gibbs = MB_gibbs.get_adj_matrix(thresh_low=thresh_low)

        # SA with exponential cooling
        BMB_SA = MB.BMB_SA(X=data, p=p, q=q, lambda_=lambda_SA,
                           burnin=int(np.floor(0.1 * 0.1 * NSCAN)),
                           NSCAN=int(np.floor(0.9*0.1 * NSCAN)),
                           NITER_SA=int(np.floor(0.9 * NSCAN)),
                           NITER_cd=int(np.floor(0.1 * 0.9 * NSCAN)), T0=1,
                           Tn=0.005,
                           DO_COPULA=True,
                           TRUE_INVCOV=inv_cov, plugin_threshold=PLUGIN_THRESHOLD)
        BMB_SA.run()
        tmp = BMB_SA.threshold_matrix(thresh_low=thresh_low)
        adj_SA = np.zeros((p+q, p+q))
        adj_SA[0:p, p:(p+q)] = np.abs(tmp) > 0

        # TRUE Adjacency matrix of the W12 block of inv_cov
        adj_orig = np.zeros((p+q, p+q))
        adj_orig[0:p, p:(p+q)] = np.abs(inv_cov[0:p, p:(p+q)]) > 0

        graph_plots.append(util_plots.create_graph_from_adj(
            adj_SA, p, q, title="Simulated Annealing", size=size))
        graph_plots.append(util_plots.create_graph_from_adj(
            adj_gibbs, p, q, title="Gibbs", size=size))
        graph_plots.append(util_plots.create_graph_from_adj(
            adj_orig, p, q, title="True Graph: {0}".format(file_name), size=size))
        # embed()
    return(graph_plots)


def main():
    ########## constants/options ######
    SEED = 142
    NSCAN = 1000
    BURNIN = int(np.floor(0.1 * NSCAN))
    lambda_ = 70
    lambda_SA = lambda_*20
    thresh_low = 0.20
    PLUGIN_THRESHOLD = 40

    ###########################
    graph_plots = compare_Gibbs_SA(
        SEED, NSCAN, BURNIN, lambda_, lambda_SA, thresh_low, PLUGIN_THRESHOLD=PLUGIN_THRESHOLD)

    output_file("./example.html")
    ncols = 3
    show(gridplot(graph_plots, ncols=ncols))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    main()
