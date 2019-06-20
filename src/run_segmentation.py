import argparse
import datetime
import logging
import sys
import time
import warnings
from os import makedirs
from os.path import abspath, normpath, basename, join, isdir, splitext

import numpy as np
from api import download_files, load_cool_files, search_opt_gamma, viz_opt_curves, viz_tads

warnings.filterwarnings("ignore")


def run_pipeline():
    """
    Whole pipeline function
    :return: nothing
    """
    parser = argparse.ArgumentParser(description='Setting parameters and paths.')
    parser.add_argument('-c_dir', '--coolfiles_dir', type=str, default='../data/coolfiles',
                        help='Path to the directory where all coolfiles store. Normally you do not need to modify this path.')
    parser.add_argument('-emt_dir', '--e_mtabs_dir', type=str, default='../data/e-mtabs',
                        help='Path to the directory where all e-mtab files store. Normally you do not need to modify this path.')
    parser.add_argument('-exp_dir', '--experiments_dir', type=str, default='../data/experiments',
                        help='Path to the directory where all experiments store. Normally you do not need to modify this path.')
    parser.add_argument('-it', '--input_type', type=str, default='coolfiles',
                        help='Type of input: url, e-mtab or coolfiles. If you will type "url" then e-mtab file will be downloaded '
                             'and then all coolfiles will be downloaded. If you will type "e-mtab" then only coolfiles '
                             'will be downloaded. If you will type "coolfiles" then nothing will be downloaded. In each '
                             'case you should specify input_path parameter. In case of "url" it should be url. In case '
                             'of "e-mtab" it should be e-mtab filepath. In case of "coolfiles" it should be path to the '
                             'directory where coolfiles of interest store.')
    parser.add_argument('-ip', '--input_path', type=str, default='../data/coolfiles/E-MTAB-4918.sdrf',
                        help='Path to the input. In case of input_type is "coolfiles" it should be path to the directory '
                             'where coolfiles of interest store. In case of input_type is "e-mtab" it should be a e-mtab '
                             'filepath. In case of input_type is "url" it should be an url to access e-mtab file.')
    parser.add_argument('-exp_name', '--experiment_name', type=str, default='',
                        help='Name of current experiment. Do not specify this parameter if you want your experiment name date-like.')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-2,
                        help='Delta for mean tad size during gamma parameter concretization.')
    parser.add_argument('-s', '--stage', type=str, default='3-4h',
                        help='Stage of development by which we will search TADs segmentation.')
    parser.add_argument('-res', '--resolution', type=int, default=5000,
                        help='Setting resolution of Hi-C map.')
    parser.add_argument('-chr', '--chromnames', type=str, default='X,2L,2R,3L,3R',
                        help='List of chromosomes of interest separated by comma.')
    parser.add_argument('-m', '--method', type=str, default='armatus',
                        help='Choose a method of segmentation. Only armatus and modularity support.')
    parser.add_argument('-g', '--grid', type=str,  default='0,5.0,0.01',
                        help='Choose a grid for gamma parameter search: three numbers separated by comma - lower bound, upper bound, step.')
    parser.add_argument('-e_mts', '--expected_mean_tad', type=int, default=120000,
                        help='Expected mean size of TADs. For Drosophila melanogaster preferable 120 Kb or 60 Kb or 30 Kb.')
    parser.add_argument('-mis', '--max_intertad_size', type=int, default=3,
                        help='Max intertad size. Recommended: 3 for armatus, 2 for modularity.')
    parser.add_argument('-mts', '--max_tad_size', type=int, default=10000,
                        help='Max TAD size.')
    parser.add_argument('-pcnt', '--percentile', type=float, default=99.9,
                        help='Percentile for cooler and Hi-C visualization.')
    parser.add_argument('-vbc', '--viz_bin_count', type=int, default=1000,
                        help='Number of bins to vizualize on a single Hi-C map.')

    args = parser.parse_args()

    COOLFILES_DIR = abspath(normpath(args.coolfiles_dir))
    if not isdir(COOLFILES_DIR): makedirs(COOLFILES_DIR)
    E_MTABS_DIR = abspath(normpath(args.e_mtabs_dir))
    if not isdir(E_MTABS_DIR): makedirs(E_MTABS_DIR)
    EXPERIMENTS_DIR = abspath(normpath(args.experiments_dir))
    if not isdir(EXPERIMENTS_DIR): makedirs(EXPERIMENTS_DIR)

    INPUT_TYPE = args.input_type
    INPUT_PATH = abspath(normpath(args.input_path)) if INPUT_TYPE != 'url' else args.input_path

    EXPERIMENT_NAME = args.experiment_name if args.experiment_name != '' else "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    E_MTAB_NAME = basename(INPUT_PATH) if INPUT_TYPE == 'coolfiles' else splitext(basename(INPUT_PATH))[0]
    EXPERIMENT_PATH = join(EXPERIMENTS_DIR, E_MTAB_NAME, EXPERIMENT_NAME)
    if not isdir(EXPERIMENT_PATH): makedirs(EXPERIMENT_PATH)

    EPSILON = args.epsilon
    STAGE_NAME = args.stage
    RESOLUTION = args.resolution
    CHROMNAMES = args.chromnames; CHROMNAMES = [x.strip() for x in CHROMNAMES.split(',')]
    METHOD = args.method
    GRID = args.grid; GRID = [float(x.strip()) for x in GRID.split(',')]
    EXPECTED_MEAN_TAD = args.expected_mean_tad
    MAX_INTERTAD = args.max_intertad_size
    MAX_TAD_SIZE = args.max_tad_size
    PERCENTILE = args.percentile
    VIZ_BIN_COUNT = args.viz_bin_count

    logging.basicConfig(filename=join(EXPERIMENT_PATH, 'segmentation.log'), filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("RUN_PIPELINE| running pipeline...")
    logging.info("RUN_PIPELINE| input command line params:\ncoolfiles_dir: {0}\ne_mtabs_dir: {1}\nexperiments_dir: {2}\n"
                 "input_type: {3}\ninput_path: {4}\nexperiment_name: {5}\ne_mtab_name: {6}\n"
                 "experiment_path: {7}\nepsilon: {8}\nstage_name: {9}\nresolution: {10}\n"
                 "chromnames: {11}\nmethod: {12}\ngrid: {13}\n"
                 "expected_mean_tad: {14}\nmax_intertad: {15}\nmax_tad_size: {16}\n"
                 "percentile: {17}\nviz_bin_count: {18}".format(COOLFILES_DIR, E_MTABS_DIR, EXPERIMENTS_DIR, INPUT_TYPE, INPUT_PATH,
                                           EXPERIMENT_NAME, E_MTAB_NAME, EXPERIMENT_PATH, EPSILON, STAGE_NAME, RESOLUTION,
                                           str(CHROMNAMES), METHOD, str(GRID), EXPECTED_MEAN_TAD, MAX_INTERTAD,
                                           MAX_TAD_SIZE, PERCENTILE, VIZ_BIN_COUNT))

    in_time = time.time()

    COOLFILES_PATH = download_files(INPUT_TYPE, INPUT_PATH, E_MTABS_DIR, COOLFILES_DIR)
    DATASETS = load_cool_files(COOLFILES_PATH, CHROMNAMES, [STAGE_NAME])
    opgs, df, df_conc = search_opt_gamma(DATASETS, EXPERIMENT_PATH, method=METHOD,
                                         grid=np.arange(GRID[0], GRID[1], GRID[2]), mis=MAX_INTERTAD, mts=MAX_TAD_SIZE,
                                         start_step=GRID[2], chrms=CHROMNAMES, eps=EPSILON, expected=EXPECTED_MEAN_TAD,
                                         exp=STAGE_NAME, resolution=RESOLUTION, percentile=PERCENTILE)
    viz_opt_curves(df, METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000), EXPERIMENT_PATH)
    viz_tads(EXPERIMENT_PATH, df_conc, DATASETS, CHROMNAMES, STAGE_NAME, method=None, clusters=False, colors=None, percentile=99.9, vbc=VIZ_BIN_COUNT)

    time_elapsed = time.time() - in_time

    logging.info("RUN_PIPELINE| whole pipeline completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    run_pipeline()
