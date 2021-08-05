import collections
import logging
import time
import warnings
from os.path import join
from collections import OrderedDict
import random
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set(context='paper', style='whitegrid')
warnings.filterwarnings("ignore")


def clusters_dynamics(df, columns, clusters):
    """
    Function to vizualize dynamics of clusters in space of stages.
    :param df: dataframe with performed clustering.
    :param columns: list of names of columns by which clustering was performed
    :param clusters: name of df column with clusters
    :return: seaborn colors palette to encode clusters with certain colors.
    """
    n_clusters = len(set(df[clusters]))
    method = clusters.split('_')[1]
    colors = sns.color_palette('rainbow', n_clusters)
    colors1 = [[max(x[0] - 0.2, 0), max(x[1] - 0.2, 0), max(x[2] - 0.2, 0)] for x in colors]

    fig, axes = plt.subplots(n_clusters, 1, sharey=True, figsize=[5, int(15 * n_clusters / 4)])

    v_pres = {}
    for v, (i, r) in zip(df[clusters], df.iterrows()):
        color = colors[v]
        if not v in v_pres.keys():
            v_pres[v] = 0
            try:
                axes[v].plot(r[columns], label=v, color=color, alpha=0.1)
            except:
                axes.plot(r[columns], label=v, color=color, alpha=0.1)
        else:
            try:
                axes[v].plot(r[columns].values, color=color, alpha=0.1)
            except:
                axes.plot(r[columns].values, color=color, alpha=0.1)
        v_pres[v] += 1

    centroids = [list(np.mean(df[df[clusters] == nk][columns])) for nk in range(n_clusters)]
    for v, c in enumerate(centroids):
        color = colors1[v]
        try:
            axes[v].plot(c, color=color, alpha=0.9, lw=2)
        except:
            axes.plot(c, color=color, alpha=0.9, lw=2)

    for v in v_pres:
        try:
            axes[v].set_xticklabels([])
            axes[v].set_title("Cluster: {} N: {}".format(v, v_pres[v]))
        except:
            axes.set_xticklabels([])
            axes.set_title("Cluster: {} N: {}".format(v, v_pres[v]))

    try:
        axes[-1].set_xticklabels(columns, rotation=90)
    except:
        axes.set_xticklabels(columns, rotation=90)

    plt.draw()
    plt.show()

    return colors


def _pca(df, columns, clusters):
    """
    Function to perform PCA (2D and 3D, if applicable) for the clustering.
    :param df: dataframe with performed clustering.
    :param columns: list of names of columns by which clustering was performed
    :param clusters: name of df column with clusters
    :return: --
    """
    n_clusters = len(set(df[clusters]))

    if len(columns) >= 3:
        pca = PCA(n_components=3)
    else:
        pca = PCA(n_components=2)

    pca_result = pca.fit_transform(df[columns].values)
    df_pca = df.copy()
    df_pca['pca-one'] = pca_result[:, 0]
    df_pca['pca-two'] = pca_result[:, 1]

    if len(columns) >= 3:
        df_pca['pca-three'] = pca_result[:, 2]

    logging.info("PLOT|PCA| Explained variation per principal component: {}".format(pca.explained_variance_ratio_))

    ax = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=clusters,
        palette=sns.color_palette("hls", n_clusters),
        data=df_pca,
        legend="full",
        alpha=0.3
    )
    plt.draw()
    plt.show()

    if len(columns) >= 3:
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
        ax.scatter(
            xs=df_pca["pca-one"],
            ys=df_pca["pca-two"],
            zs=df_pca["pca-three"],
            c=df_pca[clusters],
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')

        plt.draw()
        plt.show()


def _tsne(df, columns, clusters, perplexity=30, rs=42):
    """
    Function to perform tSNE for the clustering.
    :param df: dataframe with performed clustering.
    :param columns: list of names of columns by which clustering was performed
    :param clusters: name of df column with clusters
    :param perplexity: parameter for tSNE method.
    :param rs: random state for tSNE method.
    :return: --
    """
    n_clusters = len(set(df[clusters]))
    method = clusters.split('_')[1]

    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=rs)
    tsne_results = tsne.fit_transform(df[columns])
    logging.info("PLOT|TSNE| t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
    df_tsne = df.copy()
    df_tsne['tsne-2d-one'] = tsne_results[:, 0]
    df_tsne['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=clusters,
        palette=sns.color_palette("hls", n_clusters),
        data=df_tsne,
        legend="full",
        alpha=0.3
    )
    plt.draw()
    plt.show()


def viz_opt_curves(df, opt_df, method, chromnames, expected_mts=60000, mts=1000, resolution=5000, stage='3-4h'):
    """
    Function to vizualize curves of coverage value, mean tad size and number of tads depend on gamma values in our grid.
    :param df: for modularity and armatus -- dataframe with all segmentations based on all gamma values from our grid.
    for insulation -- dictionary with statistics.
    :param opt_df: dataframe with segmentation for optimal gamma / window values
    :param method: TAD or TAD boundaries calling method (insulation, armatus or modularity).
    :param chromnames: list of chromosomes of interest.
    :param expected_mts: expected mean size of TADs
    :param mts: maximum TAD size.
    :param resolution: resolution of Hi-C maps
    :param stage: stage of development by which TAD or TAD boundaries calling was performed
    :return: --
    """
    # expected_mts /= resolution
    for ch in chromnames:
        if method == 'insulation':
            od = collections.OrderedDict(sorted(df[ch].items()))
            gr_mean = [od[i][0] for i in od]
            # gr_cov = [od[i][1] for i in od]
            # gr_count = [od[i][2] for i in od]
            gr_ins = [od[i][3] for i in od]
            gr_bsc = [od[i][4] for i in od]
            w_range = [i for i in od]
        else:
            gr_mean = df.query('ch=="{}"'.format(ch)).groupby(['gamma', 'ch']).mean().reset_index().sort_values(['ch', 'gamma'])
            gr_count = df.query('ch=="{}"'.format(ch)).groupby(['gamma', 'ch']).count().reset_index().sort_values(
                ['ch', 'gamma'])
            gr_cov = df.query('ch=="{}"'.format(ch)).groupby(['gamma', 'ch']).sum().reset_index().sort_values(['ch', 'gamma'])

        plt.figure(figsize=[10, 5])
        host = host_subplot(111, axes_class=AA.Axes)

        par1 = host.twinx()
        par2 = host.twinx()

        offset = 70
        new_fixed_axis = par1.get_grid_helper().new_fixed_axis
        par1.axis["left"] = new_fixed_axis(loc="left", axes=par1, offset=(-offset, 0))

        offset = 120
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis["left"] = new_fixed_axis(loc="left", axes=par2, offset=(-offset, 0))

        if method == 'insulation':
            host.set_xlabel("Window")
            host.set_ylabel("Mean TAD size")
            par1.set_ylabel("Mean Insulation score")
            par2.set_ylabel("Mean B-score")
        else:
            host.set_xlabel("Gamma")
            host.set_ylabel("Mean TAD size")
            par1.set_ylabel("Coverage")
            par2.set_ylabel("TADs count")

        if method == 'insulation':
            p1, = host.plot(w_range, gr_mean * resolution, label="{} mean TAD size".format(ch))
            p1, = host.plot([min(w_range), max(w_range)], [expected_mts, expected_mts], color=p1.get_color())
            p1, = host.plot(
                [list(set(opt_df[opt_df.ch == ch]['window']))[0], list(set(opt_df[opt_df.ch == ch]['window']))[0]],
                [0, expected_mts], color=p1.get_color(), linestyle='dashed')
            p2, = par1.plot(w_range, gr_ins, label="{} mean insulation score".format(ch))
            p3, = par2.plot(w_range, gr_bsc, label="{} mean B-score".format(ch))
        else:
            p1, = host.plot(gr_mean.gamma, gr_mean.length, label="{} mean TAD size".format(ch))
            p1, = host.plot([min(df[df['ch'] == ch]['gamma']), max(df[df['ch'] == ch]['gamma'])], [expected_mts, expected_mts],
                            color=p1.get_color())
            p2, = par1.plot(gr_cov.gamma, gr_cov.length, label="{} coverage".format(ch))
            p3, = par2.plot(gr_count.gamma, gr_count.length, label="{} count".format(ch))
            host.set_ylim([0, max(gr_mean.length)])

        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["left"].label.set_color(p2.get_color())
        par2.axis["left"].label.set_color(p3.get_color())

        if method == 'insulation':
            plt.title('Stage {}, Chr {}, Method: {}, expected TAD size: {} Kb, optimal window: {}'.format(stage,
                                                                                                          ch, method, str(expected_mts), list(set(opt_df[opt_df.ch == ch]['window']))[0] // resolution))
        else:
            plt.title('Stage {}, Chr {}, Method: {}, expected TAD size: {} Kb, optimal gamma: {}'.format(stage,
                                                                                                         ch, method, str(expected_mts), list(set(opt_df[opt_df.ch == ch]['gamma']))[0] // resolution))
        plt.draw()
        plt.show()


def viz_tads(df, datasets, begin=0, end=100, ch='chrX', exp='3-4h', resolution=5000, method=None, is_insulation=False, clusters=False, colors=None, percentile=99.9):
    """
    Function to vizualize TADs or TAD boundaries on the Hi-C matrices.
    :param df: dataframe with segmentation/clustering.
    :param datasets: python dictionary with loaded chromosomes and stages.
    :param begin: start bin for visualization
    :param end: end bin for visualization
    :param ch: chromosome of interest
    :param exp: stage of development by which we want to visualize segmentation/clustering.
    :param resolution: Hi-C resolution of your coolfiles.
    :param method: clustering method. Type in case of clusters=True.
    :param is_insulation: True in case of TAD boundaries annotation in df , False in case of TAD segmentation in df
    :param clusters: True if we want to vizualize clustering, False otherwise.
    :param colors: color pallet for clustering vizualization.
    :param percentile: percentile for cooler preparations and Hi-C vizualization.
    :return: --
    """
    color_dict = {"#7bc8f6": "lightblue", "#76ff7b": "lightgreen", "#faee66": "yellowish", "#fc86aa": "pinky",  "#a8a495": "greyish", "#070d0d": "almost black", "#fd8d49": "orangeish", "#98568d": "purpleish"}

    df_tmp = df.query("ch=='{}'".format(ch))
    segments = df_tmp[['bgn', 'end']].values // resolution
    mtx_cor = datasets[exp][ch]
    np.fill_diagonal(mtx_cor, 0)
    plt.figure(figsize=[20, 20])
    sns.heatmap(np.log(mtx_cor[begin: end, begin: end] + 1), cmap="Reds", square=True, cbar=False,
                vmax=np.nanpercentile(mtx_cor, percentile))
    plt.xticks([])
    plt.yticks([])

    if clusters:
        clusters_name = '_'.join(['cluster', method])
        for l, seg in zip(df_tmp[clusters_name].values, segments):
            if is_insulation:
                if seg[0] < end and seg[1] > begin:
                    for i in range(1):
                        plt.plot([seg[0] + i - begin, seg[1] + i - begin],
                                 [seg[0] - i - begin, seg[0] - i - begin],
                                 color=colors[l], linewidth=7, label=str(l))
                        plt.plot([seg[1] + i - begin, seg[1] + i - begin],
                                 [seg[0] - i - begin, seg[1] - i - begin],
                                 color=colors[l], linewidth=7, label=str(l))
                        plt.plot([seg[0] + i - begin, seg[1] + i - begin],
                                 [seg[0] + 1 - i - begin, seg[0] + 1 - i - begin],
                                 color=colors[l], linewidth=7, label=str(l))
                        plt.plot([seg[1] - 1 + i - begin, seg[1] - 1 + i - begin],
                                 [seg[0] - i - begin, seg[1] - i - begin],
                                 color=colors[l], linewidth=7, label=str(l))
            else:
                if seg[0] < end and seg[1] > begin:
                    plt.plot([seg[0] - begin, seg[1] - begin], [seg[0] - begin, seg[0] - begin],
                             color=colors[l], linewidth=7, label=str(l))
                    plt.plot([seg[1] - begin, seg[1] - begin], [seg[0] - begin, seg[1] - begin],
                             color=colors[l], linewidth=7, label=str(l))
    else:
        for ii, seg in enumerate(segments):
            if is_insulation:
                if seg[0] < end and seg[1] > begin:
                    for i in range(1):
                        plt.plot(
                            [seg[0] + i - begin, seg[1] + i - begin],
                            [seg[0] - i - begin, seg[0] - i - begin],
                            color='blue', linewidth=7)
                        plt.plot(
                            [seg[1] + i - begin, seg[1] + i - begin],
                            [seg[0] - i - begin, seg[1] - i - begin],
                            color='blue', linewidth=7)
                        plt.plot(
                            [seg[0] + i - begin, seg[1] + i - begin],
                            [seg[0] + 1 - i - begin,
                             seg[0] + 1 - i - begin],
                            color='blue', linewidth=7)
                        plt.plot([seg[1] - 1 + i - begin,
                                  seg[1] - 1 + i - begin],
                                 [seg[0] - i - begin,
                                  seg[1] - i - begin],
                                 color='blue', linewidth=7)
            else:
                if seg[0] < end and seg[1] > begin:
                    plt.plot([seg[0] - begin, seg[1] - begin], [seg[0] - begin, seg[0] - begin], color='blue', linewidth=7)
                    plt.plot([seg[1] - begin, seg[1] - begin], [seg[0] - begin, seg[1] - begin], color='blue', linewidth=7)

    if clusters:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        lgd = plt.legend(by_label.values(), by_label.keys(), title='Clusters:', loc='upper right', prop={'size': 25})
        lgd.get_title().set_fontsize('25')

    plt.draw()
    plt.show()
