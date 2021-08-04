import logging
import time
import operator
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation, MeanShift

from hichew.hichew.lib import utils

warnings.filterwarnings("ignore")


def boundaries(matrices, coolers, label='3-4h', expected_tad_size=60000, grid=None, chromnames=None,
               max_intertad=3, max_tad=1000, percentile=99.9, eps=0.05, window_eps=5, min_dist_bad_bin=3,
               filtration='auto', bs_thresholds=None, bs_thresholds_grid=None):
    """
    Function to call TAD boundaries.
    :param matrices: python dictionary with loaded chromosomes and stages.
    :param coolers: python dictionary with cooler files that correspond to selected stages of development.
    :param label: coolfile name to call TAD boundaries. In case of developmental Hi-C data we recommend
    to use last stage of development to call both TADs and TAD boundaries
    :param expected_tad_size: TAD size to be expected in the investigated organism.
    It could be found in papers as mean / median TAD size
    :param grid: list of values for window size parameter (in bp) of insulation method.
    If None -- default grid will be selected
    :param chromnames: list of chromosomes of interest. If None -- all chromosomes will be considered.
    :param max_intertad: maximum intertad size
    :param max_tad: maximum tad size.
    :param percentile: percentile for cooler preparations and Hi-C vizualization.
    Normally should be 99.9, but you could set another value.
    :param eps: delta for stopping criterion in optimization of window search.
    Lower value gives you more accurate optimal gamma value in the end.
    :param window_eps: number of previous window value to be averaged for stopping criterion.
    :param min_dist_bad_bin: number of bins to the 'noisy' white stripes to be ommited while TAD boundaries calling
    (parameter for Cooltools insulation method)
    :param filtration: type of filtration by boundary strength score.
    If 'auto' -- thresholds on boundary strength score will be selected according to the grid specified in parameter bs_thresholds_grid;
    If 'custom' -- threshold will be selected based on the bs_thresholds parameter.
    :param bs_thresholds: thresholds for 'custom' filtration of boundary strength score.
    It is a dictionary {'label': threshold}. If None -- 'auto' mode will be selected.
    :param bs_thresholds_grid: list of threshold values for 'auto' filtration of boundary strength score.
    It is a list [0.0, 0.05, 0.10, ...] If None -- a standard grid will be selected.
    :return: python dictionary with optimal window values for each chromosome, dataframe with segmentation for all
    window values in given range, dataframe with segmentation for optimal window values and dictionary with stats for
    each chromosome.
    """
    if chromnames:
        chrms = chromnames
    else:
        chrms = list(coolers.values())[0].chromnames

    resolution = list(coolers.values())[0].binsize

    if not grid:
        grid = np.arange(0, 30, 1) * resolution

    if filtration == 'auto':
        if bs_thresholds_grid:
            bs_grid = bs_thresholds_grid
        else:
            bs_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    elif filtration == 'custom':
        bs_grid = [bs_thresholds[label]]
    else:
        logging.info("CALL|BOUNDARIES| Error: incorrect type of filtration")
        raise Exception('Please, set correct type of filtration!')

    df = pd.DataFrame(columns=['bgn', 'end', 'bs_threshold', 'window', 'ch', 'insulation_score', 'boundary_strength'])
    opt_windows = {}
    stats = {x: {} for x in chrms}

    logging.info("CALL|BOUNDARIES| Start search optimal annotation...")
    time_start = time.time()

    for ch in chrms:
        logging.info("CALL|BOUNDARIES| Start chromosome {}".format(ch))
        filters, mtx, good_bins = utils.get_noisy_stripes(matrices, ch, resolution, label, percentile=percentile, method='insulation')

        all_bins_cnt = good_bins.shape[0]
        good_bins_cnt = good_bins[good_bins == True].shape[0]
        tads_expected_cnt = good_bins_cnt // (expected_tad_size / resolution)

        logging.info("CALL|BOUNDARIES| For chromosome {} with {} bins we have {} good bins and expected count of "
                     "TADs (according to expected TAD size) of {}".format(ch, all_bins_cnt, good_bins_cnt, tads_expected_cnt))
        logging.info("CALL|BOUNDARIES| Run TAD boundaries search using windows grid for chromosome {}...".format(ch))

        df_bsc = pd.DataFrame(
            columns=['bgn', 'end', 'bs_threshold', 'window', 'ch', 'insulation_score', 'boundary_strength'])
        opt_windows_bsc = {}  # key: bs_threshold, value: opt window
        stats_bsc = {x: {} for x in bs_grid}  # key: bs_threshold, value: stats for each window

        # loop for boundary strength param
        for bsg in bs_grid:
            logging.info("CALL|BOUNDARIES| Run TAD boundaries search for boundary strength threshold {}...".format(bsg))
            boundaries_coords_0, boundaries_0 = utils.produce_boundaries_segmentation(coolers[label], grid[0], ch, min_dist_bad_bin, bsg=bsg)
            mean_tad_size_prev, sum_cov_prev, mean_ins_prev, mean_bsc_prev, full_ins_prev, full_bsc_prev = \
                utils.calc_mean_tad_size(boundaries_0, filters, ch, max_intertad, max_tad, grid[0], resolution)

            window_prev = grid[0]

            f_ins_prev = full_ins_prev
            f_bsc_prev = full_bsc_prev

            ins_prev = mean_ins_prev
            bsc_prev = mean_bsc_prev
            cov_prev = sum_cov_prev
            bound_count_prev = len(boundaries_coords_0)

            stats_bsc[bsg][grid[0]] = (mean_tad_size_prev, cov_prev, bound_count_prev, ins_prev, bsc_prev)

            df_tmp = pd.DataFrame(columns=['bgn', 'end', 'bs_threshold', 'window', 'ch', 'insulation_score',
                                           'boundary_strength'], index=np.arange(len(boundaries_coords_0)))
            df_tmp.loc[:, ['bgn', 'end']] = boundaries_coords_0
            df_tmp['bs_threshold'] = bsg
            df_tmp['window'] = grid[0]
            df_tmp['ch'] = ch
            df_tmp.loc[:, 'insulation_score'] = f_ins_prev
            df_tmp.loc[:, 'boundary_strength'] = f_bsc_prev
            df_bsc = pd.concat([df_bsc, df_tmp])

            local_optimas = {}
            mean_tad_sizes = []
            covs = []
            mean_tad_sizes.append(mean_tad_size_prev)
            covs.append(cov_prev)
            ins = []
            ins.append(ins_prev)
            bsc = []
            bsc.append(bsc_prev)
            bounds_cnt = []
            bounds_cnt.append(bound_count_prev)

            is_exp_tad_cnt_reached = False
            is_exp_tad_size_reached = False

            for window in grid[1:]:
                boundaries_coords, boundaries = utils.produce_boundaries_segmentation(coolers[label], window, ch, min_dist_bad_bin, bsg=bsg)
                mean_tad_size, sum_cov, mean_ins, mean_bsc, full_ins, full_bsc = utils.calc_mean_tad_size(boundaries, filters,
                                                                                                    ch, max_intertad, max_tad,
                                                                                                    window, resolution)

                mean_tad_sizes.append(mean_tad_size)
                cov = sum_cov
                covs.append(cov)
                ins.append(mean_ins)
                bsc.append(mean_bsc)
                bound_count = len(boundaries_coords)
                bounds_cnt.append(bound_count)

                stats_bsc[bsg][window] = (mean_tad_size, cov, bound_count, mean_ins, mean_bsc)

                df_tmp = pd.DataFrame(columns=['bgn', 'end', 'bs_threshold', 'window', 'ch', 'insulation_score',
                                               'boundary_strength'], index=np.arange(len(boundaries_coords)))
                df_tmp.loc[:, ['bgn', 'end']] = boundaries_coords
                df_tmp['window'] = window
                df_tmp['bs_threshold'] = bsg
                df_tmp['ch'] = ch
                df_tmp.loc[:, 'insulation_score'] = full_ins
                df_tmp.loc[:, 'boundary_strength'] = full_bsc
                df_bsc = pd.concat([df_bsc, df_tmp])

                if (mean_tad_size - expected_tad_size / resolution) * (mean_tad_size_prev - expected_tad_size / resolution) <= 0:
                    is_exp_tad_size_reached = True
                    if abs(mean_tad_size - expected_tad_size / resolution) <= abs(mean_tad_size_prev - expected_tad_size / resolution):
                        local_optimas[window] = cov
                    else:
                        local_optimas[window_prev] = cov_prev

                if (bound_count_prev - tads_expected_cnt) * (bound_count - tads_expected_cnt) <= 0:
                    is_exp_tad_cnt_reached = True

                if is_exp_tad_size_reached and is_exp_tad_cnt_reached and len(
                        [x for x in bounds_cnt if str(x) != 'nan']) >= window_eps:
                    window_cnts = np.asarray([x for x in bounds_cnt if str(x) != 'nan'][-window_eps:])
                    window_cnts = abs(window_cnts - tads_expected_cnt)
                    # add "and window > 150000" in the end of below if-condition if you want to limit output to the certain
                    # window value:
                    if np.mean(abs(window_cnts[:-1] - window_cnts[-1]) / window_cnts[-1]) < eps:
                        break

                window_prev = window
                cov_prev = cov
                mean_tad_size_prev = mean_tad_size
                bound_count_prev = bound_count

            local_optimas[grid[np.argmin([abs(x - expected_tad_size / resolution) for x in mean_tad_sizes])]] = covs[
                np.argmin([abs(x - expected_tad_size / resolution) for x in mean_tad_sizes])]
            opt_window = max(local_optimas.items(), key=operator.itemgetter(1))[0]
            opt_windows_bsc[bsg] = opt_window

            logging.info("CALL|BOUNDARIES| Found optimal window for chrm {} and boundary strength threshold {}: {}".format(ch, bsg, opt_window))
            _, _ = utils.produce_boundaries_segmentation(coolers[label], opt_window, ch, min_dist_bad_bin, bsg=bsg)
            logging.info("CALL|BOUNDARIES| End boundary strength threshold {}".format(bsg))

        bsc_list = []
        mts_list = []
        for bsg in bs_grid:
            bsc_list.append(stats_bsc[bsg][opt_windows_bsc[bsg]][-1])
            # print(stats_bsc[bsg][opt_windows_bsc[bsg]][0])
            mts_list.append(abs(stats_bsc[bsg][opt_windows_bsc[bsg]][0] - expected_tad_size / resolution))
        bsc_list, mts_list, bs_grid_new = zip(*sorted(zip(bsc_list, mts_list, bs_grid), reverse=True))
        bsc_list = list(bsc_list)
        mts_list = list(mts_list)
        bs_grid_new = list(bs_grid_new)

        # print(mts_list)
        # print(bsc_list)
        # print(bs_grid_new)

        for i in range(1, len(mts_list) - 1):
            if mts_list[i] < mts_list[i - 1] and mts_list[i] < mts_list[i + 1]:
                #logging.info(mts_list[i])
                if mts_list[i] < 1:  # Now less than 1 bin of Hi-C map -- should be adjusted in order to depend on expected TAD size)
                    best_index = i
                    break
        #logging.info(np.argmin(mts_list))
        #logging.info(bs_grid_new[np.argmin(mts_list)])
        # best_index = np.argmin(mts_list)
        try:
            best_boundary_strength_threshold = bs_grid_new[best_index]
        except:
            best_boundary_strength_threshold = bs_grid_new[np.argmin(mts_list)]

        if best_boundary_strength_threshold < 0.2:
            logging.warning('CALL|BOUNDARIES| WARNING! Your expected TAD size parameter value is probably too low! '
                            'The subsequent borders annotation might be incorrect! Please, choose another one (higher)')
        elif best_boundary_strength_threshold >= 0.8:
            logging.warning('CALL|BOUNDARIES| WARNING! Your expected TAD size parameter value is probably too high! '
                            'The subsequent borders annotation might be incorrect! Please, choose another one (lower)')

        logging.info('CALL|BOUNDARIES| For stage {} for chrom {} boundary strength percentile is {}'.format(label, ch,
                                                                                                              best_boundary_strength_threshold))
        # best_boundary_strength_threshold = bs_grid[np.argmax(bsc_list)]
        best_window = opt_windows_bsc[best_boundary_strength_threshold]
        stats[ch] = stats_bsc[best_boundary_strength_threshold]
        opt_windows[ch] = best_window

        sub_df = df_bsc[df_bsc.bs_threshold == best_boundary_strength_threshold]
        sub_df.index = list(range(sub_df.shape[0]))
        df = pd.concat([df, sub_df])

        logging.info("CALL|BOUNDARIES| End chromosome {}".format(ch))

    df.loc[:, 'window'] = df.window.values.astype(int)
    df.loc[:, 'bs_threshold'] = df.bs_threshold.values.astype(float)
    df.loc[:, 'bgn'] = df.bgn.values.astype(int)
    df.loc[:, 'end'] = df.end.values.astype(int)
    df.loc[:, 'insulation_score'] = df.insulation_score.values.astype(float)
    df.loc[:, 'boundary_strength'] = df.boundary_strength.values.astype(float)
    df.reset_index(drop=True, inplace=True)

    df_opt = pd.DataFrame(columns=['bgn', 'end', 'bs_threshold', 'window', 'ch', 'insulation_score', 'boundary_strength'])
    for ch in chrms:
        df_opt = pd.concat([df_opt, df[(df['ch'] == ch) & (df['window'] == opt_windows[ch])]])
    df_opt.reset_index(drop=True, inplace=True)

    time_delta = time.time() - time_start
    m, s = divmod(time_delta, 60); h, m = divmod(m, 60)
    logging.info("CALL|BOUNDARIES| Searching optimal segmentation completed in {:.0f}h {:.0f}m {:.0f}s".format(h, m, s))

    return df, df_opt, stats, opt_windows


def domains(matrices, coolers, method='armatus', label='3-4h', expected_tad_size=60000, grid=None, chromnames=None, max_intertad=3, max_tad=1000, percentile=99.9, eps=1e-2):
    if not grid:
        if method == 'armatus':
            grid = np.arange(0, 5, 0.01)
        elif method == 'modularity':
            grid = np.arange(0, 200, 0.1)

    start_step = grid[1] - grid[0]

    if chromnames:
        chrms = chromnames
    else:
        chrms = list(coolers.values())[0].chromnames

    resolution = list(coolers.values())[0].binsize

    df = pd.DataFrame(columns=['bgn', 'end', 'gamma', 'method', 'ch'])
    df_concretized = pd.DataFrame(columns=['bgn', 'end', 'gamma', 'method', 'ch'])
    opt_gammas = {}

    logging.info("CALL|DOMAINS| Start search optimal segmentation...")
    time_start = time.time()
    for ch in chrms:
        logging.info("CALL|DOMAINS| Start chromosome {}".format(ch))

        filters, mtx, good_bins = utils.get_noisy_stripes(matrices, ch, resolution, label, percentile=percentile, method=method)
        utils.whether_to_expand(mtx, filters, grid, ch, good_bins, method, max_intertad, max_tad, start_step)
        adj_grid = utils.adjust_boundaries(mtx, filters, grid, ch, good_bins, method, max_intertad, max_tad, start_step, eps=eps,
                                     type='upper')
        adj_grid = utils.adjust_boundaries(mtx, filters, adj_grid, ch, good_bins, method, max_intertad, max_tad, start_step, eps=eps,
                                     type='lower')
        df, opt_gamma = utils.find_global_optima(mtx, filters, adj_grid, ch, good_bins, method, max_intertad, max_tad, start_step, df, expected_tad_size, resolution)
        df_concretized, opt_gammas = utils.adjust_global_optima(mtx, filters, opt_gamma, opt_gammas, ch, good_bins, method, max_intertad, max_tad, start_step, df_concretized, expected_tad_size, resolution, eps=eps)

        logging.info("CALL|DOMAINS| End chromosome {}".format(ch))

    df.loc[:, 'gamma'] = df.gamma.values.astype(float)
    df.loc[:, 'bgn'] = df.bgn.values.astype(int)
    df.loc[:, 'end'] = df.end.values.astype(int)
    df.loc[:, 'length'] = df.end - df.bgn
    df.reset_index(drop=True, inplace=True)

    df_concretized.loc[:, 'gamma'] = df_concretized.gamma.values.astype(float)
    df_concretized.loc[:, 'bgn'] = df_concretized.bgn.values.astype(int)
    df_concretized.loc[:, 'end'] = df_concretized.end.values.astype(int)
    df_concretized.loc[:, 'length'] = df_concretized.end - df_concretized.bgn
    df_concretized.reset_index(drop=True, inplace=True)

    time_delta = time.time() - time_start
    m, s = divmod(time_delta, 60); h, m = divmod(m, 60)
    logging.info("CALL|DOMAINS| Searching optimal segmentation completed in {:.0f}h {:.0f}m {:.0f}s".format(h, m, s))

    return opt_gammas, df, df_concretized


def clusters(df, colnames, method='kmeans', n_clusters=6, rs=42, damping=0.7, max_iter=400, convergence_iter=15):
    """
    Function to perform clustering under the given (optimal) segmentation.
    :param df: dataframe with segmentation and calculated D-z-scores.
    :param colnames: list of names of columns by which to perform clustering
    :param seg_path: path to the file with (optimal) segmentation by which we want to perform clustering.
    :param data_path: path to the experiment's directory.
    :param mode: mode of clustering. Mode 'range' means that we want to perform clustering for a range of number of
    clusters (to select then the best number of clusters by vizual assessment). Mode 'certain' means that we want to
    launch our clustering under the certain number of clusters value.
    :param method: clustering method. Available: 'kmeans', 'meanshift', 'hierarchical', 'spectral', 'affinity_propagation'.
    :param n_clusters: in case of mode='certain' - number of clusters. In case of mode='range' - maximum number K of
    clusters in range 1..K.
    :param stages: list of developmental stages by which we want to built clustering.
    :param rs: random state for clustering/tSNE methods. Pass 0 in case you want to have no random state during your experiments.
    :param damping: damping parameter for affinity propagation clustering.
    :param max_iter: max_inter parameter for affinity propagation clustering.
    :param convergence_iter: convergence_iter parameter for affinity propagation clustering.
    :return: adjusted dataframe with clustering (add a column 'cluster_METHOD' with cluster's labels.
    """
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=rs).fit(df[colnames])
        centroids, labels_ = km.cluster_centers_, km.labels_
        df.loc[:, "cluster_kmeans"] = labels_
    elif method == 'meanshift':
        ms = MeanShift().fit(df[colnames])
        centroids, labels_ = ms.cluster_centers_, ms.labels_
        df.loc[:, "cluster_meanshift"] = labels_
    elif method == 'hierarchical':
        ac = AgglomerativeClustering(n_clusters=n_clusters).fit(df[colnames])
        labels_ = ac.labels_
        df.loc[:, "cluster_hierarchical"] = labels_
    elif method == 'spectral':
        sc = SpectralClustering(n_clusters=n_clusters, random_state=rs).fit(df[colnames])
        labels_ = sc.labels_
        df.loc[:, "cluster_spectral"] = labels_
    elif method == 'affinity_propagation':
        ap = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter).fit(df[colnames])
        centroids, labels_ = ap.cluster_centers_, ap.labels_
        df.loc[:, "cluster_affinity_propagation"] = labels_
    else:
        logging.error('CALL|CLUSTERS| Choose correct clustering method!')
        raise Exception('Choose correct clustering method!')

    return df
