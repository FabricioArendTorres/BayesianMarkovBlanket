#!/usr/bin/env python3
# coding: utf-8

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import networkx as nx

from bokeh.layouts import gridplot
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, GraphRenderer, Oval
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4, linear_palette
from bokeh.models import Title, LinearColorMapper

import numpy as np
import pandas as pd
from BMB import MB


def create_graph_from_nxop(adj1, adj2, p, q, nxop=nx.intersection, title="", subtitle="",  size=500):
    assert(callable(nxop))

    G1 = nx.from_pandas_adjacency(pd.DataFrame(adj1))
    G2 = nx.from_pandas_adjacency(pd.DataFrame(adj2))

    G_op = nxop(G1, G2)
    return(create_graph_from_invcov(nx.to_numpy_matrix(G_op), p, q, title, subtitle, size, show_weights=False))

def find_nearest(array, value, idx=False):
    idx = (np.abs(array - value)).argmin()
    return(idx)

def nx_falsenegatives(G1, G2):
    return(nx.difference(nx.complement(G1), nx.complement(G2)))

def best_perf_graph(data, size, dataid, inv_cov_MB):
    rows_of_interest = [#find_nearest(data.specificity, 0.99),
                        # find_nearest(data.sensitivity, 0.9),
                        find_nearest(data["zero_edges/total_edges"], 0.95),
                        #np.argmax(data.precision),
                        np.argmax(data.f_score),
                        np.argmax(data.MCC)
                        ]
    titles = [#"Spec closest to 0.99: {0}".format(np.round(data.specificity[find_nearest(data.specificity, 0.99)],3)),
              #"Sens closest to 0.9: {0}".format(np.round(data.sensitivity[find_nearest(data.sensitivity, 0.9)],3)),
              "Sparsity: {0}".format(np.round(data.specificity[find_nearest(data["zero_edges/total_edges"], 0.95)],3)),
              #"Max Precision: "+str(np.round(np.max(data.precision),3)),
              "Max F Score: "+str(np.round(np.max(data.f_score), 3)),
              "Max MCC: "+str(np.round(np.max(data.MCC), 3))
              ]

    graph_plots = []
    for i, title in zip(rows_of_interest, titles):
        w12 = np.loadtxt(open(data["W12Path"][i], "rb"),
                         delimiter=",", skiprows=0)
        # util_plots.create_graph_from_adj()
        p = w12.shape[0]
        q = w12.shape[1]
        assert(p == data["p"][i] and q == data["q"][i])
        adj = np.zeros((p+q, p+q))
        adj[0:p, p:(p+q)] = w12
        graph_plots.append(create_graph_from_invcov(adj, p=p, q=q, size=size,
                                                    title=title, subtitle="ConfigID:" +
                                                    str(data["ConfigID"][i]) +
                                                    " Lambda:" +
                                                    str(data["lambda"][i])
                                                    + " ITER:" +
                                                    str(data["NSCAN"][i])
                                                    + "/" +
                                                    str(data["NITER"][i])))
        # graph_plots.append(create_graph_from_invcov(inv_cov_MB, p=p, q=q, size=size, title="True Graph",
        #                                             subtitle="DataID: "+str(dataid)+" p:"+str(p)+" q:"+str(q)))
        
        graph_plots.append(create_graph_from_nxop(adj, inv_cov_MB, p=p, q=q, nxop=nx.intersection,
                                                  size=size, title="True Positives",
                                                  subtitle="DataID: "+str(dataid)+" p:"+str(p)+" q:"+str(q)))
        # graph_plots.append(create_graph_from_nxop(adj, inv_cov_MB, p=p, q=q, nxop=nx.difference,
        #                                           size=size, title="False Positives",
        #                                           subtitle="DataID: "+str(dataid)+" p:"+str(p)+" q:"+str(q)))
        graph_plots.append(create_graph_from_nxop(adj, inv_cov_MB, p=p, q=q, nxop=nx_falsenegatives,
                                                  size=size, title="False Negatives",
                                                  subtitle="DataID: "+str(dataid)+" p:"+str(p)+" q:"+str(q)))
    return(graph_plots)


def create_networkx_from_invcov(invcov, remove_nodes=True):
    """Creates networkx graph from inverse covariance matrix and returns a bokeh plot of it.

    Arguments:
        invcov {np.ndarray} -- Inverse Covariance matrix of a grpah

    Keyword Arguments:
        title {str} -- Optional Title of the plot (default: {""})
    """
    assert(invcov.shape[0] == invcov.shape[1])
    invcov = pd.DataFrame(invcov)

    G = nx.from_pandas_adjacency(invcov)

    if(remove_nodes):
        G.remove_nodes_from(list(nx.isolates(G)))
    return(G)


def create_graph_from_invcov(invcov, p, q, title="", subtitle="",  size=500, remove_nodes=True, labels=None, show_weights=True):
    """Creates networkx graph from inverse covariance matrix and returns a bokeh plot of it.

    Arguments:
        invcov {np.ndarray} -- Inverse Covariance matrix of a grpah

    Keyword Arguments:
        title {str} -- Optional Title of the plot (default: {""})
    """
    assert(invcov.shape[0] == invcov.shape[1])
    invcov = pd.DataFrame(invcov)

    G = nx.from_pandas_adjacency(invcov)

    if(remove_nodes):
        G.remove_nodes_from(list(nx.isolates(G)))

    graph_renderer = from_networkx(
        G, nx.shell_layout, nlist=[list(range(p)), list(range(p, p+q))])

    graph_renderer.node_renderer.data_source.add(
        [(k < p)*max(k, 1) for k, v in G.degree()], 'is_p')
    graph_renderer.node_renderer.data_source.add(
        [min(10+v, 20) for k, v in G.degree()], 'degree')
    # mapper = LinearColorMapper(palette=linear_palette(Spectral, p+1), low=0, high=p)
    mapper = LinearColorMapper(
        palette=linear_palette(Spectral4, 2), low=0, high=1)

    graph_renderer.node_renderer.glyph = Circle(
        size='degree', fill_color={'field': 'is_p', 'transform': mapper})
    graph_renderer.node_renderer.selection_glyph = Circle(
        size='degree', fill_color=Spectral4[3])
    graph_renderer.node_renderer.hover_glyph = Circle(
        size='degree', fill_color=Spectral4[3])

    edge_mapper = LinearColorMapper(
        palette=['pink', 'palegreen'], low=-1, high=1)

    edge_vals = [G.get_edge_data(u, v)["weight"] for u, v in G.edges]
    edge_weights = [
        min(max(abs(G.get_edge_data(u, v)["weight"]), 0.5)*4, 6) for u, v in G.edges]
    graph_renderer.edge_renderer.data_source.add(edge_weights, 'weights')
    graph_renderer.edge_renderer.data_source.add(np.sign(edge_vals), 'signs')

    if(show_weights):
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color={'field': 'signs', 'transform': edge_mapper}, line_alpha=1.0, line_width='weights')
    else:
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#CCCCCC", line_alpha=1.0, line_width=1)
    
    graph_renderer.edge_renderer.selection_glyph = MultiLine(
        line_color='black', line_width='weights')
    graph_renderer.edge_renderer.hover_glyph = MultiLine(
        line_color="#CCCCCC", line_width='weights')

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    if(labels is not None):
        tmp = [labels[i] for i in G.nodes]
        graph_renderer.node_renderer.data_source.data['name'] = tmp
        neighbours = []
        adj = np.array(np.abs(invcov) > 0)
        neighbours = [
            labels[np.where(np.logical_or(adj[i, :], adj[:, i]))] for i in G.nodes]
                        
        graph_renderer.node_renderer.data_source.data['neighbours'] = neighbours

        tooltips = [("idx:", "@index"),
                    ("Name:", "@name"),
                    ("Neighbours:", "@neighbours{safe}")
                    ]
    else:
        neighbours = []
        adj = np.array(np.abs(invcov) > 0)
        for i in list(G.nodes):
            neighbours.append(list(map(str, np.where(np.logical_or(adj[i, :], adj[:, i]))[0])))
        graph_renderer.node_renderer.data_source.data['neighbours'] = neighbours
        tooltips = [("idx:", "@index"),
                    ("Neighbours:", "@neighbours{safe}")]
                    
    # add line breaks
    for n in neighbours:
        if len(n)>3:
            for i in range(len(n)):
                if(i%3==0):
                    n[i]="<br/>"+n[i]
                    
    bokeh_pl = Plot(plot_width=size, plot_height=size,
                    x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    # bokeh_pl.title.text = title
    bokeh_pl.add_layout(
        Title(text=subtitle, text_font_style="italic"), 'above')
    bokeh_pl.add_layout(Title(text=title, text_font_size="16pt"), 'above')

    # bokeh_pl.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
    hover = HoverTool(tooltips=tooltips)
    bokeh_pl.add_tools(hover, TapTool(), BoxSelectTool())

    bokeh_pl.renderers.append(graph_renderer)

    return(bokeh_pl)


def create_graph_from_adj(adj, p, q, title="", subtitle="",  size=500, remove_nodes=True):
    """Creates networkx graph from adjacency matrix and returns a bokeh plot of it.

    Arguments:
        adj {np.ndarray} -- Adjacency matrix of a grpah

    Keyword Arguments:
        title {str} -- Optional Title of the plot (default: {""})
    """
    assert(adj.shape[0] == adj.shape[1])

    G = nx.from_numpy_matrix(adj)
    if(remove_nodes):
        G.remove_nodes_from(list(nx.isolates(G)))

    graph_renderer = from_networkx(
        G, nx.shell_layout, nlist=[list(range(p)), list(range(p, p+q))])

    graph_renderer.node_renderer.data_source.add(
        [(k < p)*max(k, 1) for k, v in G.degree()], 'is_p')
    graph_renderer.node_renderer.data_source.add(
        [min(10+v, 20) for k, v in G.degree()], 'degree')
    # mapper = LinearColorMapper(palette=linear_palette(Spectral, p+1), low=0, high=p)
    mapper = LinearColorMapper(
        palette=linear_palette(Spectral4, 2), low=0, high=1)

    graph_renderer.node_renderer.glyph = Circle(
        size='degree', fill_color={'field': 'is_p', 'transform': mapper})
    graph_renderer.node_renderer.selection_glyph = Circle(
        size='degree', fill_color=Spectral4[3])
    graph_renderer.node_renderer.hover_glyph = Circle(
        size='degree', fill_color=Spectral4[3])

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#CCCCCC", line_alpha=1.0, line_width=3)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(
        line_color=Spectral4[2], line_width=3)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(
        line_color=Spectral4[1], line_width=3)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()
    neighbours = []
    for i in list(G.nodes):
        neighbours.append(np.where(np.logical_or(adj[i, :], adj[:, i])))
    graph_renderer.node_renderer.data_source.data['neighbours'] = neighbours

    bokeh_pl = Plot(plot_width=size, plot_height=size,
                    x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    # bokeh_pl.title.text = title
    bokeh_pl.add_layout(
        Title(text=subtitle, text_font_style="italic"), 'above')
    bokeh_pl.add_layout(Title(text=title, text_font_size="16pt"), 'above')

    # bokeh_pl.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
    hover = HoverTool(tooltips=[("idx:", "@index"),
                                ("Neighbours:", "@neighbours")])
    bokeh_pl.add_tools(hover, TapTool(), BoxSelectTool())

    bokeh_pl.renderers.append(graph_renderer)

    return(bokeh_pl)


def running_mean(vals):
    running_mean = np.zeros(len(vals)+1)
    for n in range(1, len(vals)+1):
        running_mean[n] = running_mean[n-1] + (vals[n-1]-running_mean[n-1])/n
    return(running_mean)


def diagnostics_fig(W_list, idx_row, idx_col, size=(15,15), suptitle="", matrix_name="W",notitle=False, r_mean=True, l_width=1):
    """Create diagnostice figure for one specific entry of the inverse covariance matrix.
        Figures:
        - Trace plot
        - Density Plot
        - Running Mean
        - ACF

    Arguments:
        W_list {[type]} -- [description]
        idx_row {[type]} -- [description]
        idx_col {[type]} -- [description]
    
    Keyword Arguments:
        size {int} -- [description] (default: {0})
        suptitle {str} -- [description] (default: {""})
    """

    sns.set_style("darkgrid")
    data_var = pd.Series(W_list[:, idx_row, idx_col])
    plt.figure(figsize=size)
    if(r_mean):
        gs = gridspec.GridSpec(3, 3)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, :])
        ax3 = plt.subplot(gs[2, :-1])
        ax4 = plt.subplot(gs[2, 2])
    else:
        gs = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(gs[0, :])
        ax3 = plt.subplot(gs[1, :-1])
        ax4 = plt.subplot(gs[1, 2])
    # ax5 = plt.subplot(gs[0:2, 2])
    if(not notitle):
        plt.suptitle(matrix_name+"["+str(idx_row)+","+str(idx_col)+"] " + suptitle, fontsize = 24)

    plt.sca(ax1)
    plt.title("Trace")
    plt.plot(data_var, linewidth=0.5*l_width)
    # plt.xlabel("Accepted Sample #")

    if(r_mean):
        plt.sca(ax2)
        plt.title("Running Mean")

        plt.plot(running_mean(data_var), linewidth=l_width)
        # plt.xlabel("Accepted Sample #")

    plt.sca(ax3)
    plt.title("ACF")
    pd.plotting.autocorrelation_plot(data_var, ax=ax3, linewidth=0.5*l_width)

    plt.sca(ax4)
    plt.title("Density")
    sns.distplot(data_var)
    plt.ylabel("density")

    # plt.sca(ax5)
    # adj[idx_row, idx_col] = -1
    # adj[idx_col, idx_row] = -1

    # mask = np.zeros_like(adj, dtype=np.bool)
    # mask[np.tril_indices_from(mask)] = True
    # sns.heatmap(adj, mask=mask,
    #             center=0, square=True, linewidths=.5, vmax=1, vmin=-1, cbar_kws={"shrink": .5})

    plt.tight_layout(rect=[0, 0.09, 1, 0.95])
    plt.subplots_adjust(top=0.8)
    fig = plt.gcf()
    return(fig)
