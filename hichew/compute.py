import logging
import time
import warnings

import cooltools
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from hichew.hichew.lib import utils

warnings.filterwarnings("ignore")


def normalize(df, columns, type_norm='z-score-row'):
    """
    type_norm: Type of normalization (z-score-row, z-score-col, min-max-col, min-max-row, log-col, log-row)
    """
    df_copy = df.copy()
    for col in columns:
        df_copy.loc[:, 'norm_{}'.format(col)] = 0
    if type_norm == 'z-score-row':
        df_copy[['norm_{}'.format(col) for col in columns]] = np.array([x for x in df_copy.loc[:, columns].apply(scipy.stats.zscore, axis=1).values])
        df_copy = df_copy.dropna(axis=0, subset=['norm_{}'.format(col) for x in columns]).reset_index(drop=True)
    elif type_norm == 'z-score-col':
        df_copy[['norm_{}'.format(col) for col in columns]] = np.array([x for x in df_copy.loc[:, columns].apply(scipy.stats.zscore, axis=0).values])
        df_copy = df_copy.dropna(axis=0, subset=['norm_{}'.format(col) for x in columns]).reset_index(drop=True)
    elif type_norm == 'min-max-col':
        scaler = MinMaxScaler((-1, 1))
        df_copy[['norm_{}'.format(col) for col in columns]] = scaler.fit_transform(np.asarray(df_copy.loc[:, columns]))
        df_copy = df_copy.dropna(axis=0, subset=['norm_{}'.format(col) for x in columns]).reset_index(drop=True)
    elif type_norm == 'min-max-row':
        scaler = MinMaxScaler((-1, 1))
        df_copy[['norm_{}'.format(col) for col in columns]] = scaler.fit_transform(np.asarray(df_copy.loc[:, columns]).T).T
        df_copy = df_copy.dropna(axis=0, subset=['norm_{}'.format(col) for x in columns]).reset_index(drop=True)
    elif type_norm == 'log-col':
        df_copy = df_copy.dropna(axis=0, subset=columns).reset_index(drop=True)
        ins_arr = np.asarray(df_copy.loc[:, columns])
        df_copy[['norm_{}'.format(col) for col in columns]] = np.log(ins_arr - np.min(ins_arr) + 1)
    elif type_norm == 'log-row':
        df_copy = df_copy.dropna(axis=0, subset=columns).reset_index(drop=True)
        ins_arr = np.asarray(df_copy.loc[:, columns])
        ins_arr_new = np.asarray([np.log(x - np.min(x) + 1) for x in ins_arr])
        df_copy[['norm_{}'.format(col) for col in columns]] = ins_arr_new

    return df_copy


def d_scores(df, matrices, stages):
    """
    Function to compute D-z-scores to perform clustering.
    :return: adjusted dataframe with D-scores columns for each stage.
    """
    logging.info("COMPUTE|D_SCORES| Start computing D-scores...")

    in_time = time.time()
    df_res = pd.DataFrame()
    chrms = list(set(df.ch))

    for ch in chrms:
        df_tmp = df.query("ch=='{}'".format(ch))
        if df_tmp.shape[0] == 0: continue
        segments = df_tmp[['bgn', 'end']].values
        for exp in stages:
            mtx_cor = matrices[exp][ch]
            np.fill_diagonal(mtx_cor, 0)
            Ds = utils.get_d_score(mtx_cor, segments)
            df_tmp.loc[:, "D_{}".format(exp)] = Ds

        df_tmp.reset_index(drop=True)
        df_res = df_res.append(df_tmp, ignore_index=True)
        df_res = df_res.dropna(axis=0).reset_index(drop=True)
    time_elapsed = time.time() - in_time
    logging.info(
        "COMPUTE|D_SCORES| Complete computing D-scores in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    return df_res


def silhouette(df, columns, clusters):
    """
    Function to get silhouette score of our clustering
    :param df: dataframe with performed clustering.
    :return: silhouette score
    """
    try:
        return silhouette_score(df[columns], list(df[clusters]))
    except:
        logging.info("COMPUTE|SILHOUETTE_SCORE| WARNING! CAN'T CALCULATE SILHOUETTE SCORE. IT SEEMS THAT YOU HAVE ONLY 1 CLUSTER.")
        return 0.0


def insulation_scores(df, coolers, stages, chromnames=None, ignore_diags=2):
    """
    Function to compute Insulation-z-scores to perform clustering. Only for method=insulation usage!
    :param seg_path: path to the file with final (optimal) segmentation.
    :param cool_sets: python dictionary with cooler files that correspond to selected stages of development.
    :param stages: list of developmental stages.
    :param chrms: list of chromosomes.
    :param ignore_diags: parameter for cooltools calculate_insulation_score method.
    :return: adjusted dataframe with insulation-z-scores columns for each stage.
    """
    logging.info("COMPUTE|INSULATION_SCORES| Start computing insulation scores...")
    in_time = time.time()

    if chromnames:
        chrms = chromnames
    else:
        chrms = list(coolers.values())[0].chromnames

    for stage in stages:
        ins_scores = pd.DataFrame(
            columns=['chrom', 'start', 'end', 'is_bad_bin', 'log2_insulation_score', 'n_valid_pixels'])
        for ch in chrms:
            opt_window_ch = df.query("ch=='{}'".format(ch))['window'].iloc[0]
            sub_df = cooltools.insulation.calculate_insulation_score(coolers[stage], int(opt_window_ch),
                                                                     ignore_diags=ignore_diags, chromosomes=[ch])
            sub_df.rename(columns={'log2_insulation_score_{}'.format(int(opt_window_ch)): 'log2_insulation_score',
                                   'n_valid_pixels_{}'.format(int(opt_window_ch)): 'n_valid_pixels'}, inplace=True)
            ins_scores = pd.concat([ins_scores, sub_df])
        ins_scores.reset_index(drop=True, inplace=True)
        df['ins_score_{}'.format(stage)] = list(map(lambda x, y, z: ins_scores[
            (ins_scores['start'] == x) & (ins_scores['end'] == y) & (ins_scores['chrom'] == z)][
            'log2_insulation_score'].iloc[-1], df['bgn'], df['end'], df['ch']))

    segmentation = df.dropna(axis=0, subset=['ins_score_{}'.format(x) for x in stages]).reset_index(drop=True)
    time_elapsed = time.time() - in_time
    logging.info(
        "COMPUTE|INSULATION_SCORES| Complete computing insulation scores in {:.0f}m {:.0f}s".format(time_elapsed // 60,
                                                                                               time_elapsed % 60))
    return segmentation
