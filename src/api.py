import glob
import logging
import time
import warnings
from os import listdir, makedirs
from os.path import basename, join, isdir, splitext

import cooler
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import numpy as np
import pandas as pd
import requests
import scipy
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from utils import whether_to_expand, get_noisy_stripes, adjust_boundaries, find_global_optima, adjust_global_optima, \
	get_d_score

sns.set(context='paper', style='whitegrid')
warnings.filterwarnings("ignore")


def download_files(input_type, input_path, e_mtabs_dir, coolfiles_dir):
	"""
	Function to download E-MTAB file and/or Coolfiles.
	:param input_type: type of input (url, coolfiles or e-mtab).
	:param input_path: path to the input (url, path to the directory with coolfiles, path to the e-mtab file).
	:param e_mtabs_dir: path to the directory where all E_MTAB files store.
	:param coolfiles_dir: path to the directory where all coolfiles store.
	:return: path to the directory with coolfiles.
	"""
	if input_type == 'coolfiles':
		logging.info("DOWNLOAD_FILES| Your coolfiles are located in directory {}".format(input_path))
		return input_path
	elif input_type == 'e-mtab' or input_type == 'url':
		if input_type == 'url':
			r = requests.get(input_path, allow_redirects=True)
			e_mtab_path = join(e_mtabs_dir, basename(input_path))
			open(e_mtab_path, 'wb').write(r.content)
			logging.info("DOWNLOAD_FILES| E-MTAB file {} has been downloaded in directory {} by access url {}".format(
				basename(input_path), e_mtabs_dir, input_path))
		else:
			logging.info("DOWNLOAD_FILES| E-MTAB file {} is located in directory {}".format(basename(input_path),
																							e_mtabs_dir))
			e_mtab_path = input_path
		logging.info("DOWNLOAD_FILES| Reading E-MTAB file {}".format(e_mtab_path))
		data = pd.read_csv(e_mtab_path, sep='\t')
		cool_files = list(set(data[data['Comment[experiment]'] == 'Hi-C']['Derived Array Data File']))
		baseurl = "https://www.ebi.ac.uk/arrayexpress/files/"
		urls = [(baseurl +
				 data[data['Derived Array Data File'] == x].iloc[0]['Comment [Derived ArrayExpress FTP file]'].split('/')[-2] + '/' +
				 data[data['Derived Array Data File'] == x].iloc[0]['Comment [Derived ArrayExpress FTP file]'].split('/')[-1] + '/' + x)
				for x in cool_files]
		coolfiles_path = join(coolfiles_dir, splitext(basename(input_path))[0])
		logging.info("DOWNLOAD_FILES| All coolfiles will be downloaded in directory {}".format(coolfiles_path))
		if not isdir(coolfiles_path):
			logging.info("DOWNLOAD_FILES| Make directory {}".format(coolfiles_path))
			makedirs(coolfiles_path)
		if len(listdir(coolfiles_path)) == 0:
			logging.info("DOWNLOAD_FILES| Directory {} is empty. Let's download our coolfiles!".format(coolfiles_path))
			logging.info("DOWNLOAD_FILES| Start downloading coolfiles...")
			in_time = time.time()
			for url, file in zip(urls, cool_files):
				filename = join(coolfiles_path, file)
				r = requests.get(url, allow_redirects=True)
				open(filename, 'wb').write(r.content)
			time_elapsed = time.time() - in_time
			logging.info("DOWNLOAD_FILES| Downloading completed in {:.0f}m {:.0f}s".format(time_elapsed // 60,
																						   time_elapsed % 60))

		return coolfiles_path


def load_cool_files(coolfiles_path, chromnames, stage_names=None):
	"""
	Function to load coolfiles.
	:param coolfiles_path: path to the directory with coolfiles.
	:param chromnames: list of chromosomes names.
	:param stage_names: name(s) of developmental stage to investigate. In case of clustering we recommend to pass None
	to load all stages you have.
	:return: python dictionary with keys of stages and chromnames, and values of contact matrices.
	"""
	if not stage_names is None:
		files = [x for x in glob.glob(join(coolfiles_path, '*.cool')) if any(sn == '_'.join(basename(x).split('_')[0:-3]) for sn in stage_names)]
	else:
		files = [x for x in glob.glob(join(coolfiles_path, '*.cool'))]
	logging.info("LOAD_COOL_FILES| List of coolfiles of interest: {}".format(str(files)))
	labels = ['_'.join(x.split('/')[-1].split('_')[0: -3]) for x in files]
	datasets = {x: {} for x in labels}
	balance = True
	logging.info("LOAD_COOL_FILES| Start loading chromosomes and stages...")
	in_time = time.time()
	for label, file in list(zip(labels, files)):
		c = cooler.Cooler(file)
		for ch in chromnames:
			mtx = c.matrix(balance=balance).fetch(ch)
			datasets[label][ch] = mtx.copy()
	time_elapsed = time.time() - in_time
	logging.info("LOAD_COOL_FILES| Loading completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

	return datasets


def search_opt_gamma(datasets, experiment_path, method, grid, mis, mts, start_step, chrms, eps=1e-2, expected=120000,
					 exp='3-4h', resolution=5000, percentile=99.9):
	"""
	Function to search optimal gamma for each chromosome
	:param datasets: python dictionary with loaded chromosomes and stages.
	:param experiment_path: path to experiment directory.
	:param method: segmentation method (only armatus and modularity available).
	:param grid: range for optimal gamma value search.
	:param mis: maximum intertad size corresponds to the method (armatus or modularity).
	:param mts: maximum tad size.
	:param start_step: start step to search optimal gamma value. It is equal to step in grid param.
	:param chrms: list of chromosomes of interest.
	:param eps: delta for mean tad size during gamma search. Normally equal to 1e-2.
	Lower values gives you more accurate optimal gamma value in the end.
	:param expected: tad size to be expected. For Drosophila melanogaster it could be 120000, 60000 or 30000 bp.
	:param exp: stage of development by which we will search TADs segmentation. It should be single stage.
	Normally -- last one, by which TADs are getting their formation.
	For Drosophila melanogaster this stage should be 3-4h or nuclear_cycle_14 in case you have not 3-4h.
	:param resolution: Hi-C resolution of your coolfiles.
	:param percentile: percentile for cooler preparations and Hi-C vizualization.
	Normally should be 99.9, but you could set another value.
	:return: python dictionary with optimal gamma values for each chromosome, dataframe with segmentation for all
	gamma values in given range, dataframe with segmentation for optimal gamma values.
	"""
	df = pd.DataFrame(columns=['bgn', 'end', 'gamma', 'method', 'ch'])
	df_concretized = pd.DataFrame(columns=['bgn', 'end', 'gamma', 'method', 'ch'])
	opt_gammas = {}

	logging.info("SEARCH_OPT_GAMMA| Start search optimal segmentation...")
	time_start = time.time()
	for ch in chrms:
		logging.info("SEARCH_OPT_GAMMA| Start chromosome {}".format(ch))

		filters, mtx, good_bins = get_noisy_stripes(datasets, ch, exp=exp, percentile=percentile)
		whether_to_expand(mtx, filters, grid, ch, good_bins, method, mis, mts, start_step)
		adj_grid = adjust_boundaries(mtx, filters, grid, ch, good_bins, method, mis, mts, start_step, eps=eps,
									 type='upper')
		adj_grid = adjust_boundaries(mtx, filters, adj_grid, ch, good_bins, method, mis, mts, start_step, eps=eps,
									 type='lower')
		df, opt_gamma = find_global_optima(mtx, filters, adj_grid, ch, good_bins, method, mis, mts, start_step, df, expected, resolution)
		df_concretized, opt_gammas = adjust_global_optima(mtx, filters, opt_gamma, opt_gammas, ch, good_bins, method, mis, mts, start_step, df_concretized, expected, resolution, eps=eps)

		logging.info("SEARCH_OPT_GAMMA| End chromosome {}".format(ch))

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

	df.to_csv(join(experiment_path, "all_tads_{0}_{1}kb_{2}kb.csv".format(method, int(expected / 1000),
																		  int(resolution / 1000))), sep='\t')
	df_concretized.to_csv(join(experiment_path, "opt_tads_{0}_{1}kb_{2}kb.csv".format(method, int(expected / 1000),
																					  int(resolution / 1000))), sep='\t')

	logging.info("SEARCH_OPT_GAMMA| Write optimal segmentation in file {}".format(join(experiment_path,
					"opt_tads_{0}_{1}kb_{2}kb.csv".format(method, int(expected / 1000), int(resolution / 1000)))))

	time_delta = time.time() - time_start
	m, s = divmod(time_delta, 60); h, m = divmod(m, 60)
	logging.info("SEARCH_OPT_GAMMA| Searching optimal segmentation completed in {:.0f}h {:.0f}m {:.0f}s".format(h, m, s))

	return opt_gammas, df, df_concretized


def viz_opt_curves(df, method, chromnames, expected_mts, mts, data_path):
	"""
	Function to vizualize curves of coverage value, mean tad size and number of tads depend on gamma values in our grid.
	:param df: dataframe with all segmentations based on all gamma values from our grid.
	:param method: segmentation method (only armatus and modularity available).
	:param chromnames: list of chromosomes of interest.
	:param expected_mts: expected mean size of TADs. For Drosophila melanogaster preferable 120 Kb or 60 Kb or 30 Kb.
	:param mts: maximum TAD size.
	:param data_path: path to experiment's directory.
	:return: nothing
	"""
	for ch in chromnames:
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

		host.set_xlabel("Gamma")
		host.set_ylabel("Mean")
		par1.set_ylabel("Coverage")
		par2.set_ylabel("Count")

		p1, = host.plot(gr_mean.gamma, gr_mean.length, label="{} mean".format(ch))
		p1, = host.plot([min(df[df['ch'] == ch]['gamma']), max(df[df['ch'] == ch]['gamma'])], [expected_mts, expected_mts],
						color=p1.get_color())
		p2, = par1.plot(gr_cov.gamma, gr_cov.length, label="{} coverage".format(ch))
		p3, = par2.plot(gr_count.gamma, gr_count.length, label="{} count".format(ch))

		host.set_ylim([0, max(gr_mean.length)])

		host.axis["left"].label.set_color(p1.get_color())
		par1.axis["left"].label.set_color(p2.get_color())
		par2.axis["left"].label.set_color(p3.get_color())

		plt.title(ch + ': ' + method + ': ' + str(mts) + 'Kb')

		plt.draw()
		plt.savefig(join(data_path, ch + '_' + method + '_' + str(mts) + 'Kb' + '.png'))


def viz_tads(data_path, df, datasets, chromnames, exp, method=None, clusters=False, colors=None, percentile=99.9, vbc=1000):
	"""
	Function to vizualize TADs on our Hi-C matrix.
	:param data_path: path to experiment's directory.
	:param df: dataframe with segmentation/clustering to vizualize.
	:param datasets: python dictionary with loaded chromosomes and stages.
	:param chromnames: list of chromosomes of interest.
	:param exp: stage of development by which we vizualize segmentation/clustering.
	:param method: clustering method. Type in case of clusters=True.
	:param clusters: True if we want to vizualize clustering, False otherwise.
	:param colors: color pallet for clustering vizualization.
	:param percentile: percentile for cooler preparations and Hi-C vizualization.
	:param vbc: resolution of each chromosome segment to vizualize. You can change it in order to have desired scale
	of your vizualization.
	Value vbc=1000 means that we split our chromosome into 1000-bins-sized regions and vizualize each of them.
	:return: nothing.
	"""
	for ch in chromnames:
		mtx_size = datasets[exp][ch].shape[0]
		begin_arr = list(np.arange(0, mtx_size, vbc))
		end_arr = begin_arr[1:]; end_arr.append(mtx_size)
		for begin, end in zip(begin_arr, end_arr):
			df_tmp = df.query("ch=='{}'".format(ch))
			segments = df_tmp[['bgn', 'end']].values
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
					if seg[0] < end and seg[1] > begin:
						plt.plot([seg[0] - begin, seg[1] - begin], [seg[0] - begin, seg[0] - begin], color=colors[l])
						plt.plot([seg[1] - begin, seg[1] - begin], [seg[0] - begin, seg[1] - begin], color=colors[l])
			else:
				for seg in segments:
					if seg[0] < end and seg[1] > begin:
						plt.plot([seg[0] - begin, seg[1] - begin], [seg[0] - begin, seg[0] - begin], color='green')
						plt.plot([seg[1] - begin, seg[1] - begin], [seg[0] - begin, seg[1] - begin], color='green')

			plt.title(ch + '; ' + exp + '; ' + str(begin) + ':' + str(end) + '; ' + 'clustering: ' + str(clusters))
			plt.draw()
			plt.savefig(join(data_path, ch + '_' + exp + '_' + str(begin) + '_' + str(end) + '_' + 'clustering_' + str(clusters) + '.png'))


def compute_d_z_scores(seg_path, datasets, chrms):
	"""
	Function to copmute D-z-scores to perform clustering.
	:param seg_path: path to the file with final (optimal) segmentation.
	:param datasets: python dictionary with loaded chromosomes and stages.
	:param chrms: list of chromosomes of interest.
	:return: adjusted dataframe with D-scores columns for each stage.
	"""
	logging.info("COMPUTE_D_Z_SCORES| Start computing D-scores...")
	in_time = time.time()
	df = pd.read_csv(seg_path, sep='\t', index_col=0)
	df_res = pd.DataFrame()
	for ch in chrms:
		df_tmp = df.query("ch=='{}'".format(ch))
		if df_tmp.shape[0] == 0: continue
		segments = df_tmp[['bgn', 'end']].values
		for exp in list(datasets.keys()):
			mtx_cor = datasets[exp][ch]
			np.fill_diagonal(mtx_cor, 0)
			Ds = get_d_score(mtx_cor, segments)
			df_tmp.loc[:, "D_{}".format(exp)] = Ds

		koi = ["D_{}".format(exp) for exp in list(datasets.keys())]
		for x in koi:
			df_tmp.loc[:, 'z{}'.format(x)] = 0

		df_tmp[['z{}'.format(x) for x in koi]] = np.array([x for x in df_tmp.loc[:, koi].apply(scipy.stats.zscore, axis=1).values])
		df_tmp.reset_index(drop=True)
		df_res = df_res.append(df_tmp, ignore_index=True)
		df_res = df_res.dropna(axis=0).reset_index(drop=True)
	time_elapsed = time.time() - in_time
	logging.info(
		"COMPUTE_D_Z_SCORES| Complete computing D-scores in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
	return df_res


def viz_stats(data_path, stages, df):
	"""
	Function to vizualize some statistics based on D-z-scores.
	:param data_path: path to the experiment directory.
	:param stages: list of stages of development we want to investigate.
	:param df: adjusted dataframe with segmentation and calculated D-z-scores.
	:return: nothing.
	"""
	plt.figure(figsize=[10, 7])
	v = df[["D_{}".format(x) for x in stages]].values
	sns.distplot(v[np.isfinite(v)], label='D-score', bins=np.arange(-2, 2, 0.05))

	v = df[["zD_{}".format(x) for x in stages]].values
	sns.distplot(v[np.isfinite(v)], label='zD-score', bins=np.arange(-2, 2, 0.05))

	plt.legend()

	plt.title('D-scores and zD-scores density')
	plt.draw()
	plt.savefig(join(data_path, 'scores_density.png'))

	sns.clustermap(df.loc[:, ["zD_{}".format(x) for x in stages]].corr(), cmap='RdBu_r', center=0).savefig(join(data_path, 'stages_correlation.png'))

	colors = sns.color_palette('Set1', len(["zD_{}".format(x) for x in stages]))

	fig, axes = plt.subplots(len(["zD_{}".format(x) for x in stages]), 1, sharey=True, figsize=[5, 15])

	v_pres = {}
	for i, r in df.iterrows():
		v = np.argmax(r[["zD_{}".format(x) for x in stages]].values)
		color = colors[v]
		if not v in v_pres.keys():
			v_pres[v] = 0
			axes[v].plot(r[["zD_{}".format(x) for x in stages]], label=v, color=color, alpha=0.5)
		else:
			axes[v].plot(r[["zD_{}".format(x) for x in stages]].values, color=color, alpha=0.2)
		v_pres[v] += 1

	for v in v_pres:
		axes[v].set_xticklabels([])
		axes[v].set_title("{}; {} TADs".format(["zD_{}".format(x) for x in stages][v], v_pres[v]))

	axes[-1].set_xticklabels(["zD_{}".format(x) for x in stages], rotation=90)

	plt.draw()
	plt.savefig(join(data_path, 'tads_stages_dynamics.png'))


def perform_clustering(df, seg_path, data_path, mode, method, n_clusters, stages, rs, damping, max_iter, convergence_iter):
	"""
	Function to perform clustering under the given (optimal) segmentation.
	:param df: adjusted dataframe with segmentation and calculated D-z-scores.
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
	if mode == 'range':
		if method == 'kmeans':
			sum_of_squared_distances = []
			for k in range(1, n_clusters + 1):
				km = KMeans(n_clusters=k, random_state=rs)
				km = km.fit(df[["zD_{}".format(x) for x in stages]])
				sum_of_squared_distances.append(km.inertia_)
			plt.figure(figsize=[10, 7])
			plt.plot(np.arange(1, n_clusters + 1), sum_of_squared_distances, 'bx-')
			plt.xlabel('k')
			plt.ylabel('Sum_of_squared_distances')
			plt.title('Elbow Method For Optimal k')
			plt.draw()
			plt.savefig(join(data_path, 'elbow_kmeans.png'))
		elif method == 'meanshift' or method == 'affinity_propagation':
			logging.error('PERFORM_CLUSTERING| Use "certain" mode for {} method!'.format(method))
			return
		elif method == 'hierarchical' or method == 'spectral':
			# fig, ax1 = plt.subplots(n_clusters, 1, sharey=True, figsize=[15, int(15 * n_clusters / 4)])
			for n_cluster in range(2, n_clusters + 1):
				fig, ax1 = plt.subplots(1, 1)
				fig.set_size_inches(18, 7)
				ax1.set_xlim([-0.1, 1])
				ax1.set_ylim([0, df.shape[0] + (n_cluster + 1) * 10])

				ac = AgglomerativeClustering(n_clusters=n_cluster) if method == 'hierarchical' else SpectralClustering(n_clusters=n_cluster, random_state=rs)
				ac = ac.fit(df[["zD_{}".format(x) for x in stages]])
				cluster_labels = ac.labels_

				silhouette_avg = silhouette_score(df[["zD_{}".format(x) for x in stages]], cluster_labels)

				logging.info("PERFORM_CLUSTERING| For n_clusters = {} the average silhouette_score is {}".format(n_cluster, silhouette_avg))

				sample_silhouette_values = silhouette_samples(df[["zD_{}".format(x) for x in stages]], cluster_labels)

				y_lower = 10
				for i in range(n_cluster):
					ith_cluster_silhouette_values = \
						sample_silhouette_values[cluster_labels == i]

					ith_cluster_silhouette_values.sort()

					size_cluster_i = ith_cluster_silhouette_values.shape[0]
					y_upper = y_lower + size_cluster_i

					color = cm.nipy_spectral(float(i) / n_cluster)
					ax1.fill_betweenx(np.arange(y_lower, y_upper),
									  0, ith_cluster_silhouette_values,
									  facecolor=color, edgecolor=color, alpha=0.7)

					ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

					y_lower = y_upper + 10

				ax1.set_title("The silhouette plot for the various clusters.")
				ax1.set_xlabel("The silhouette coefficient values")
				ax1.set_ylabel("Cluster label")
				ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

				ax1.set_yticks([])
				ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

				plt.suptitle(("Silhouette analysis for {} clustering on sample data with n_clusters = {}".format(method, n_clusters)),
							 fontsize=14, fontweight='bold')

				plt.draw()
				plt.savefig(join(data_path, 'silhouette_{}_nk_{}.png'.format(method, n_cluster)))
		else:
			logging.error('PERFORM_CLUSTERING| Choose correct clustering method!'.format(method))
			return
		return
	elif mode == 'certain':
		if method == 'kmeans':
			km = KMeans(n_clusters=n_clusters, random_state=rs).fit(df[["zD_{}".format(x) for x in stages]])
			centroids, labels_ = km.cluster_centers_, km.labels_
			df.loc[:, "cluster_kmeans"] = labels_
		elif method == 'meanshift':
			ms = MeanShift().fit(df[["zD_{}".format(x) for x in stages]])
			centroids, labels_ = ms.cluster_centers_, ms.labels_
			df.loc[:, "cluster_meanshift"] = labels_
		elif method == 'hierarchical':
			ac = AgglomerativeClustering(n_clusters=n_clusters).fit(df[["zD_{}".format(x) for x in stages]])
			labels_ = ac.labels_
			df.loc[:, "cluster_hierarchical"] = labels_
		elif method == 'spectral':
			sc = SpectralClustering(n_clusters=n_clusters, random_state=rs).fit(df[["zD_{}".format(x) for x in stages]])
			labels_ = sc.labels_
			df.loc[:, "cluster_spectral"] = labels_
		elif method == 'affinity_propagation':
			ap = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter).fit(df[["zD_{}".format(x) for x in stages]])
			centroids, labels_ = ap.cluster_centers_, ap.labels_
			df.loc[:, "cluster_affinity_propagation"] = labels_
		else:
			logging.error('PERFORM_CLUSTERING| Choose correct clustering method!'.format(method))
			return
		df.to_csv(join(data_path, splitext(basename(seg_path))[0] + '_clustering_{}.csv'.format(method)), sep='\t')
		return df


def viz_clusters_dynamics(df, data_path, method, stages):
	"""
	Function to vizualize dynamics of clusters in space of stages.
	:param df: dataframe with performed clustering.
	:param data_path: path to the experiment's directory.
	:param method: clustering method.
	:param stages: list of stages to investigate clusters' dynamics.
	:return: seaborn colors palette to encode clusters with certain colors.
	"""
	clusters_name = '_'.join(['cluster', method])
	n_clusters = len(set(df[clusters_name]))
	colors = sns.color_palette('rainbow', n_clusters)
	colors1 = [[max(x[0] - 0.2, 0), max(x[1] - 0.2, 0), max(x[2] - 0.2, 0)] for x in colors]

	fig, axes = plt.subplots(1, 1, sharey=True, figsize=[5, int(15 * 1 / 4)])

	v_pres = {}
	v = 0
	for i, r in df.iterrows():
		if not v in v_pres.keys():
			v_pres[v] = 0
			axes.plot(r[["zD_{}".format(x) for x in stages]], label=v, color='grey', alpha=0.1)
		else:
			axes.plot(r[["zD_{}".format(x) for x in stages]].values, color='black', alpha=0.1)
		v_pres[v] += 1

	axes.set_xticklabels(["zD_{}".format(x) for x in stages], rotation=90)
	plt.title('All clusters. Method: {}'.format(method))
	plt.draw()
	plt.savefig(join(data_path, 'all_clusters_{}.png'.format(method)))

	fig, axes = plt.subplots(n_clusters, 1, sharey=True, figsize=[5, int(15 * n_clusters / 4)])

	v_pres = {}
	for v, (i, r) in zip(df[clusters_name], df.iterrows()):
		color = colors[v]
		if not v in v_pres.keys():
			v_pres[v] = 0
			axes[v].plot(r[["zD_{}".format(x) for x in stages]], label=v, color=color, alpha=0.1)
		else:
			axes[v].plot(r[["zD_{}".format(x) for x in stages]].values, color=color, alpha=0.1)
		v_pres[v] += 1

	centroids = [list(np.mean(df[df[clusters_name] == nk][["zD_{}".format(x) for x in stages]])) for nk in range(n_clusters)]
	for v, c in enumerate(centroids):
		color = colors1[v]
		axes[v].plot(c, color=color, alpha=0.9, lw=2)

	for v in v_pres:
		axes[v].set_xticklabels([])
		axes[v].set_title("Cluster: {} N: {}".format(v, v_pres[v]))

	axes[-1].set_xticklabels(["zD_{}".format(x) for x in stages], rotation=90)

	plt.draw()
	plt.savefig(join(data_path, 'clusters_detalization_{}.png'.format(method)))

	return colors


def viz_pca(df, data_path, stages, method):
	"""
	Function to vizualize PCA of our clustering.
	:param df: dataframe with performed clustering.
	:param data_path: path to the experiment's directory.
	:param stages: list of stages to investigate clusters' dynamics.
	:param method: clustering method.
	:return: nothing.
	"""
	clusters_name = '_'.join(['cluster', method])
	n_clusters = len(set(df[clusters_name]))
	pca = PCA(n_components=3)
	pca_result = pca.fit_transform(df[["zD_{}".format(x) for x in stages]].values)
	df_pca = df.copy()
	df_pca['pca-one'] = pca_result[:, 0]
	df_pca['pca-two'] = pca_result[:, 1]
	df_pca['pca-three'] = pca_result[:, 2]
	logging.info("VIZ_PCA| Explained variation per principal component: {}".format(pca.explained_variance_ratio_))

	ax = plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="pca-one", y="pca-two",
		hue=clusters_name,
		palette=sns.color_palette("hls", n_clusters),
		data=df_pca,
		legend="full",
		alpha=0.3
	)
	plt.draw()
	plt.savefig(join(data_path, '2D_pca_{}.png'.format(method)))

	ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
	ax.scatter(
		xs=df_pca["pca-one"],
		ys=df_pca["pca-two"],
		zs=df_pca["pca-three"],
		c=df_pca[clusters_name],
		cmap='tab10'
	)
	ax.set_xlabel('pca-one')
	ax.set_ylabel('pca-two')
	ax.set_zlabel('pca-three')

	plt.draw()
	plt.savefig(join(data_path, '3D_pca_{}.png'.format(method)))


def viz_tsne(df, data_path, stages, method, perplexity, rs):
	"""
	Function to vizualize tSNE of our clustering.
	:param df: dataframe with performed clustering.
	:param data_path: path to the experiment's directory.
	:param stages: list of stages to investigate clusters' dynamics.
	:param method: clustering method.
	:param perplexity: parameter for tSNE method.
	:param rs: random state for tSNE method.
	:return: nothing.
	"""
	clusters_name = '_'.join(['cluster', method])
	n_clusters = len(set(df[clusters_name]))
	time_start = time.time()
	tsne = TSNE(n_components=2, perplexity=perplexity, random_state=rs)
	tsne_results = tsne.fit_transform(df[["zD_{}".format(x) for x in stages]])
	logging.info("VIZ_TSNE| t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
	df_tsne = df.copy()
	df_tsne['tsne-2d-one'] = tsne_results[:, 0]
	df_tsne['tsne-2d-two'] = tsne_results[:, 1]
	plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue=clusters_name,
		palette=sns.color_palette("hls", n_clusters),
		data=df_tsne,
		legend="full",
		alpha=0.3
	)
	plt.draw()
	plt.savefig(join(data_path, 't-SNE_{}.png'.format(method)))


def get_silhouette_score(df, stages, method):
	"""
	Function to get silhouette score of our clustering
	:param df: dataframe with performed clustering.
	:param stages: list of stages to investigate clusters' dynamics.
	:param method: clustering method.
	:return: silhouette score
	"""
	clusters_name = '_'.join(['cluster', method])
	return silhouette_score(df[["zD_{}".format(x) for x in stages]], list(df[clusters_name]))
