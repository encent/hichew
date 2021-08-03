import argparse
import datetime
import logging
import sys
import time
import warnings
from os import makedirs
from os.path import abspath, normpath, isdir, splitext, exists, basename, join

import seaborn as sns
from api import download_files, load_cool_files, compute_d_z_scores, viz_tads, viz_stats, \
    perform_clustering, viz_clusters_dynamics, viz_pca, viz_tsne, get_silhouette_score, compute_ins_z_scores

sns.set(context='paper', style='whitegrid')
warnings.filterwarnings("ignore")


def run_pipeline():
    """
    Whole pipeline function
    :return: nothing
    """
    parser = argparse.ArgumentParser(description='Setting parameters and paths.')
    parser.add_argument('-sp', '--segmentation_path', type=str,
                        default='../data/experiments/E-MTAB-4918.sdrf/opt_tads_modularity_60kb_5kb.csv',
                        help='Path to the file with optimal segmentation.')
    parser.add_argument('-it', '--input_type', type=str, default='coolfiles',
                        help='Type of input: url, e-mtab or coolfiles. If you will type "url" then e-mtab file will be '
                             'downloaded and then all coolfiles will be downloaded. If you will type "e-mtab" then '
                             'only coolfiles will be downloaded. If you will type "coolfiles" then nothing will be '
                             'downloaded. In each case you should specify input_path parameter. In case of "url" it '
                             'should be url. In case of "e-mtab" it should be e-mtab filepath. In case of "coolfiles" '
                             'it should be path to the directory where coolfiles of interest store.')
    parser.add_argument('-ip', '--input_path', type=str, default='../data/coolfiles/E-MTAB-4918.sdrf',
                        help='Path to the input. In case of input_type is "coolfiles" it should be path to the '
                             'directory where coolfiles of interest store. In case of input_type is "e-mtab" it should '
                             'be a e-mtab filepath. In case of input_type is "url" it should be an url to access '
                             'e-mtab file.')
    parser.add_argument('-e', '--experiment', type=str, default='',
                        help='Name of current experiment. Do not specify this parameter in case you want date-like '
                             'experiment name.')
    parser.add_argument('-mode', '--mode', type=str, default='range',
                        help='Mode to train clustering method: "range" in case you want to train clustering method on '
                             'several cluster number values, and select then optimal number of clusters by yourself '
                             '(visual methods), "certain" in case you want to train clustering method on single '
                             'cluster number value. We recommend to train first your cluster method on "--mode range" '
                             'with given "--n_clusters N" param, where N is maximum clusters number param to build '
                             'range from 1..N, then select optimal cluster number value from range '
                             '1..N by yourself (visual method), and finally train your clustering method with '
                             '--n_clusters K with "--mode certain", where K is the optimal number of clusters you have '
                             'to select during visual assessment (Elbow or Silhouette method). Meanshift and affinity_'
                             'propogation does not imply --mode range, lauch them on --mode ceratin! It is because of '
                             'they have to search optimal number of clusters themselves.')
    parser.add_argument('-m', '--method', type=str, default='kmeans',
                        help='Type here a mehod of clustering you want to perform. Available methods: kmeans, '
                             'meanshift, hierarchical, spectral, affinity_propagation.')
    parser.add_argument('-nc', '--n_clusters', type=int, default=10,
                        help='Type here maximum value of cluster number in case of using --mode range (then range '
                             '1..n_clusters will built). In case of --mode certain type here the certain number of '
                             'clusters to train clustering method of your choice.')
    parser.add_argument('-s', '--stages', type=str, default='nuclear_cycle_12_repl_merged_5kb, '
                                                            'nuclear_cycle_13_repl_merged_5kb, '
                                                            'nuclear_cycle_14_repl_merged_5kb, '
                                                            '3-4h_repl_merged_5kb',
                        help='Name of coolfiles (**without** extension) corresponding to the stages of development '
                             'by which TAD clustering will be run.')
    parser.add_argument('-chr', '--chromnames', type=str, default='X,2L,2R,3L,3R',
                        help='List of chromosomes of interest separated by comma.')
    parser.add_argument('-pcnt', '--percentile', type=float, default=99.99,
                        help='Percentile for cooler and Hi-C visualization.')
    parser.add_argument('-vbc', '--viz_bin_count', type=int, default=1000,
                        help='Number of bins to vizualize on a single Hi-C map.')
    parser.add_argument('-rs', '--random_state', type=int, default=42,
                        help='Choose random state for clustering and t-SNE methods. If you want to pass None - type 0.')
    parser.add_argument('-damping', '--damping', type=float, default=0.7,
                        help='Argument for affinity propagation clustering.')
    parser.add_argument('-max_iter', '--max_iter', type=int, default=400,
                        help='Argument for affinity propagation clustering.')
    parser.add_argument('-convergence_iter', '--convergence_iter', type=int, default=15,
                        help='Argument for affinity propagation clustering.')
    parser.add_argument('-perplexity', '--perplexity', type=int, default=30,
                        help='Argument for t-SNE algorithm.')
    parser.add_argument('-vs', '--visual_stage', type=str, default='3-4h_repl_merged_5kb',
                        help='Name of coolfile (with or without extension) corresponding to developmental stage by '
                             'which we visualize our clustering.')
    parser.add_argument('-ins', '--is_insulation', type=bool, default=False,
                        help='True if input segmentation is based on insulation score.')
    parser.add_argument('-res', '--resolution', type=int, default=5000,
                        help='Setting resolution of Hi-C map.')

    args = parser.parse_args()

    SEGMENTATION_PATH = abspath(normpath(args.segmentation_path))
    if not exists(SEGMENTATION_PATH): return

    INPUT_TYPE = args.input_type
    INPUT_PATH = abspath(normpath(args.input_path)) if INPUT_TYPE != 'url' else args.input_path

    EXPERIMENT_NAME = args.experiment if args.experiment != '' else "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    EXPERIMENT_PATH = join('../data/experiments', EXPERIMENT_NAME)
    if not isdir(EXPERIMENT_PATH): makedirs(EXPERIMENT_PATH)

    MODE = args.mode
    METHOD = args.method
    N_CLUSTERS = args.n_clusters
    STAGE_NAMES = args.stages; STAGE_NAMES = [x.strip() for x in STAGE_NAMES.split(',')]
    CHROMNAMES = args.chromnames; CHROMNAMES = [x.strip() for x in CHROMNAMES.split(',')]
    PERCENTILE = args.percentile
    VIZ_BIN_COUNT = args.viz_bin_count
    RS = args.random_state if args.random_state != 0 else None
    DAMPING = args.damping
    MAX_ITER = args.max_iter
    CONVERGENCE_ITER = args.convergence_iter
    PERPLEXITY = args.perplexity
    VISUAL_STAGE = splitext(args.visual_stage)[0]
    IS_INSULATION = args.is_insulation
    RESOLUTION = args.resolution

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=join(EXPERIMENT_PATH, 'clustering.log'), filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("RUN_PIPELINE| running pipeline...")
    logging.info("RUN_PIPELINE| input command line params:\nsegmentation_path: {}\ninput_type: {}\ninput_path: {}\n"
                 "experiment_name: {}\nexperiment_path: {}\nmode: {}\nmethod: {}\nn_clusters: {}\n"
                 "stage_names: {}\nchromnames: {}\npercentile: {}\nviz_bin_count: {}\nrandom_state: {}\n"
                 "damping: {}\nmax_iter: {}\nconvergence_iter: {}\nperplexity: {}\nvisual_stage: {}\n"
                 "is_insulation: {}\nresolution: {}".format(
        SEGMENTATION_PATH, INPUT_TYPE, INPUT_PATH, EXPERIMENT_NAME, EXPERIMENT_PATH, MODE, METHOD, N_CLUSTERS,
        str(STAGE_NAMES), str(CHROMNAMES), PERCENTILE, VIZ_BIN_COUNT, RS, DAMPING,
        MAX_ITER, CONVERGENCE_ITER, PERPLEXITY, VISUAL_STAGE, IS_INSULATION, RESOLUTION))

    in_time = time.time()

    COOLFILES_PATH = download_files(INPUT_TYPE, INPUT_PATH)
    DATASETS, COOL_SETS = load_cool_files(COOLFILES_PATH, CHROMNAMES, RESOLUTION, None)
    if IS_INSULATION:
        DATA_W_D_SCORES = compute_ins_z_scores(SEGMENTATION_PATH, COOL_SETS, list(COOL_SETS.keys()), CHROMNAMES, ignore_diags=2)
    else:
        DATA_W_D_SCORES = compute_d_z_scores(SEGMENTATION_PATH, DATASETS, CHROMNAMES)
    viz_stats(EXPERIMENT_PATH, STAGE_NAMES, DATA_W_D_SCORES, IS_INSULATION)
    FINAL_CLUSTERING = perform_clustering(DATA_W_D_SCORES, SEGMENTATION_PATH, EXPERIMENT_PATH, MODE, METHOD,
                                          N_CLUSTERS, STAGE_NAMES, RS, DAMPING, MAX_ITER, CONVERGENCE_ITER, IS_INSULATION)
    if MODE == 'certain':
        colors = viz_clusters_dynamics(FINAL_CLUSTERING, EXPERIMENT_PATH, METHOD, STAGE_NAMES, IS_INSULATION)
        viz_pca(FINAL_CLUSTERING, EXPERIMENT_PATH, STAGE_NAMES, METHOD, IS_INSULATION)
        viz_tsne(FINAL_CLUSTERING, EXPERIMENT_PATH, STAGE_NAMES, METHOD, PERPLEXITY, RS, IS_INSULATION)
        #viz_tads(EXPERIMENT_PATH, FINAL_CLUSTERING, DATASETS, CHROMNAMES, VISUAL_STAGE, RESOLUTION, METHOD, is_insulation=IS_INSULATION, clusters=True, colors=colors,
        #         percentile=PERCENTILE, vbc=VIZ_BIN_COUNT)
        SCORE = get_silhouette_score(FINAL_CLUSTERING, STAGE_NAMES, METHOD, IS_INSULATION)
        logging.info("RUN_PIPELINE| silhouette score for clustering is {}".format(SCORE))

    time_elapsed = time.time() - in_time

    logging.info("RUN_PIPELINE| whole pipeline completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    run_pipeline()
