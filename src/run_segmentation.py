import argparse
import datetime
import logging
import sys
import time
import warnings
import json
from os import makedirs
from os.path import abspath, normpath, join, isdir, splitext, exists

import numpy as np
from api import download_files, load_cool_files, search_opt_gamma, viz_opt_curves, viz_tads, search_opt_window, \
    run_consensus

warnings.filterwarnings("ignore")


def run_pipeline():
    """
    Whole pipeline function
    :return: nothing
    """
    parser = argparse.ArgumentParser(description='Setting parameters and paths.')
    parser.add_argument('-it', '--input_type', type=str, default='coolfiles',
                        help='Type of input: url, e-mtab or coolfiles. If you will type "url" then e-mtab file will be '
                             'downloaded and then all coolfiles will be downloaded. If you will type "e-mtab" then '
                             'only coolfiles will be downloaded. If you will type "coolfiles" then nothing will be '
                             'downloaded. In each case you should specify input_path parameter. In case of "url" it '
                             'should be url. In case of "e-mtab" it should be e-mtab filepath. In case of "coolfiles" '
                             'it should be path to the directory where coolfiles of interest store.')
    parser.add_argument('-ip', '--input_path', type=str, default='../data/coolfiles',
                        help='Path to the input. In case of input_type is "coolfiles" it should be path to the '
                             'directory where coolfiles of interest store. In case of input_type is "e-mtab" it should '
                             'be a e-mtab filepath. In case of input_type is "url" it should be an url to access '
                             'e-mtab file.')
    parser.add_argument('-e', '--experiment', type=str, default='',
                        help='Name of current experiment. Do not specify this parameter in case you want date-like '
                             'experiment name.')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-2,
                        help='Delta for mean tad size during gamma parameter concretization in case of method=armatus '
                             'or method=modularity. Delta for stopping criterion in case of method=insulation (0.05 recommended).')
    parser.add_argument('-s', '--stage', type=str, default='3-4h_repl_merged_5kb',
                        help='Name of coolfile(s) (with or without extension) corresponding to the stage of development '
                             'by which TADs optimal segmentation search will be run.')
    parser.add_argument('-f', '--filtration', type=str, default='auto',
                        help='Type of boundaries filtration: auto or custom')
    parser.add_argument('-bstp', '--bs_thresholds_path', type=str, default='../data/boundary_strength_thresholds.json',
                        help='Path to the file with boundary strength thresholds '
                             '(specify in case of filtration == custom). '
                             'Place name of coolfiles without extension as keys of json '
                             'and boundary strength thresholds as values (0 to 1, where 0 means no filtration and '
                             '1 means the strongest filtration (no boundaries)')
    parser.add_argument('-bstgp', '--bs_thresholds_grid_path', type=str, default='../data/boundary_strength_thresholds_grid.json',
                        help='Path to the file with boundary strength thresholds grid '
                             '(specify in case of filtration == auto). '
                             'Place python-like list of threshold values into this file, e.g. [0.1, 0.3, 0.5]')
    parser.add_argument('-c', '--consensus', type=bool, default=False,
                        help='If True -- segmentation runs for all stages indicated in stage argument; '
                             'if False -- segmentation runs for one stage indicated in stage argument')
    parser.add_argument('-mb', '--merge_boundaries', type=bool, default=False,
                        help="Use only if consensus==True! "
                             "If True -- boundaries within the same loci among different stages are merged into one "
                             "single 'mean' boundary; if False -- neighbour boundaries aren't merged into one")
    parser.add_argument('-res', '--resolution', type=int, default=5000,
                        help='Setting resolution of Hi-C map.')
    parser.add_argument('-chr', '--chromnames', type=str, default='X,2L,2R,3L,3R',
                        help='List of chromosomes of interest separated by comma.')
    parser.add_argument('-m', '--method', type=str, default='armatus',
                        help='Choose a method of segmentation. Only "armatus", "modularity" and "insulation" support '
                             '("insulation" for TADs borders detection, "modularity" and "armatus" for TADs bodies '
                             'detection).')
    parser.add_argument('-g', '--grid', type=str,  default='0,5.0,0.01',
                        help='In case of method=armatus or method=modularity -- choose a grid for gamma parameter '
                             'search: three numbers separated by comma - lower bound, upper bound, step.'
                             'In case of method=insulation -- choose a grid for window parameter search: also these '
                             'three numbers. '
                             'Note that in case of method=insulation third number -- step -- should be integer.')
    parser.add_argument('-e_mts', '--expected_mean_tad', type=int, default=60000,
                        help='Expected mean size of TADs (in base pairs).'
                             'For Drosophila melanogaster preferable 60 Kb.')
    parser.add_argument('-mis', '--max_intertad_size', type=int, default=3,
                        help='Max intertad size (in bins). Recommended: 3 for armatus and insulation, 2 for modularity.')
    parser.add_argument('-mts', '--max_tad_size', type=int, default=1000,
                        help='Max TAD size (in bins).')
    parser.add_argument('-pcnt', '--percentile', type=float, default=99.9,
                        help='Percentile for cooler and Hi-C visualization.')
    parser.add_argument('-vbc', '--viz_bin_count', type=int, default=1000,
                        help='Number of bins to vizualize on a single Hi-C map.')
    parser.add_argument('-vts', '--viz_tad_stage', type=str, default='3-4h',
                        help='Stage for TAD visualizing')
    parser.add_argument('-ns', '--noise_shift', type=str, default=3,
                        help='Interval over the noisy white stripes which will be ingored during boundaries calling')
    parser.add_argument('-loc_size', '--loc_size', type=int, default=2,
                        help='Total length of subset of boundaries to merge. In case of merge_boundaries==True')
    parser.add_argument('-nstm', '--num_stages_to_merge', type=int, default=3,
                        help='Number of stages to merge into consensus one. In case of merge_boundaries==True')

    args = parser.parse_args()

    INPUT_TYPE = args.input_type
    INPUT_PATH = abspath(normpath(args.input_path)) if INPUT_TYPE != 'url' else args.input_path

    EXPERIMENT_NAME = args.experiment if args.experiment != '' else "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    EXPERIMENT_PATH = join('../data/experiments', EXPERIMENT_NAME)
    if not isdir(EXPERIMENT_PATH): makedirs(EXPERIMENT_PATH)

    EPSILON = args.epsilon
    STAGE_NAME = args.stage
    FILTRATION = args.filtration
    THRESHOLDS = args.bs_thresholds_path
    THRESHOLDS_GRID = args.bs_thresholds_grid_path
    CONSENSUS = args.consensus
    MERGE_BOUNDARIES = args.merge_boundaries
    RESOLUTION = args.resolution
    CHROMNAMES = args.chromnames; CHROMNAMES = [x.strip() for x in CHROMNAMES.split(',')]
    METHOD = args.method
    GRID = args.grid; GRID = [float(x.strip()) for x in GRID.split(',')]
    EXPECTED_MEAN_TAD = args.expected_mean_tad
    MAX_INTERTAD = args.max_intertad_size
    MAX_TAD_SIZE = args.max_tad_size
    PERCENTILE = args.percentile
    VIZ_BIN_COUNT = args.viz_bin_count
    VIZ_TAD_STAGE = args.viz_tad_stage
    NOISE_SHIFT = args.noise_shift
    LOC_SIZE = args.loc_size
    NUM_STAGES_TO_MERGE = args.num_stages_to_merge

    if METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) > 2 and \
            len(np.arange(GRID[0], GRID[1], GRID[2])) < 10:
        WINDOW_SPOTLIGHT = 2
    elif METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) >= 10:
        WINDOW_SPOTLIGHT = 5
    elif METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) <= 2:
        raise Exception('Your grid is too small for "insulation" method!')
    elif METHOD != 'insulation':
        WINDOW_SPOTLIGHT = None

    if exists(THRESHOLDS):
        with open(THRESHOLDS) as f:
            THRESHOLDS_DATA = json.load(f)
    else:
        THRESHOLDS_DATA = None
    if exists(THRESHOLDS_GRID):
        with open(THRESHOLDS_GRID) as f:
            THRESHOLDS_GRID_DATA = json.load(f)
    else:
        THRESHOLDS_GRID_DATA = None

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=join(abspath(normpath(EXPERIMENT_PATH)), 'segmentation.log'), filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("RUN_PIPELINE| running pipeline...")
    logging.info("RUN_PIPELINE| input command line params:\n"
                 "input_type: {}\ninput_path: {}\nexperiment_name: {}\n"
                 "experiment_path: {}\nepsilon: {}\nstage_name: {}\nfiltration: {}\nthresholds: {}\n"
                 "thresholds_grid: {}\nconsensus: {}\nmerge_boundaries: {}\n"
                 "resolution: {}\nchromnames: {}\nmethod: {}\ngrid: {}\n"
                 "expected_mean_tad: {}\nmax_intertad: {}\nmax_tad_size: {}\npercentile: {}\nviz_bin_count: {}\n"
                 "window_spotlight: {}\nviz_tad_stage: {}\nnoise_shift: {}\nloc_size: {}\n"
                 "num_stages_to_merge: {}".format(INPUT_TYPE, INPUT_PATH,
                                            EXPERIMENT_NAME, EXPERIMENT_PATH, EPSILON, STAGE_NAME, FILTRATION,
                                                  str(THRESHOLDS_DATA), str(THRESHOLDS_GRID_DATA), CONSENSUS,
                                            MERGE_BOUNDARIES, RESOLUTION, str(CHROMNAMES), METHOD, str(GRID),
                                            EXPECTED_MEAN_TAD, MAX_INTERTAD, MAX_TAD_SIZE, PERCENTILE, VIZ_BIN_COUNT,
                                            WINDOW_SPOTLIGHT, VIZ_TAD_STAGE, NOISE_SHIFT, LOC_SIZE, NUM_STAGES_TO_MERGE))

    in_time = time.time()

    COOLFILES_PATH = download_files(INPUT_TYPE, INPUT_PATH)
    if not CONSENSUS:
        DATASETS, COOL_SETS = load_cool_files(COOLFILES_PATH, CHROMNAMES, RESOLUTION, [STAGE_NAME])
    else:
        DATASETS, COOL_SETS = load_cool_files(COOLFILES_PATH, CHROMNAMES, RESOLUTION, [x.strip() for x in STAGE_NAME.split(',')])
    if METHOD == 'insulation':
        if not CONSENSUS:
            df, df_conc, stats, opws = search_opt_window(DATASETS, COOL_SETS, EXPERIMENT_PATH,
                                                         grid=np.arange(GRID[0], GRID[1], GRID[2]) * RESOLUTION, mis=MAX_INTERTAD,
                                                         mts=MAX_TAD_SIZE, chrms=CHROMNAMES, method=METHOD,
                                                         resolution=RESOLUTION, expected=EXPECTED_MEAN_TAD, exp=STAGE_NAME,
                                                         percentile=PERCENTILE, eps=EPSILON,
                                                         window_eps=WINDOW_SPOTLIGHT, k=NOISE_SHIFT,
                                                         filtration=FILTRATION, bs_thresholds=THRESHOLDS_DATA,
                                                         bs_thresholds_grid=THRESHOLDS_GRID_DATA)
        else:
            df, df_conc, stats, opws = run_consensus(DATASETS, COOL_SETS, EXPERIMENT_PATH,
                                                         grid=np.arange(GRID[0], GRID[1], GRID[2]) * RESOLUTION, mis=MAX_INTERTAD,
                                                         mts=MAX_TAD_SIZE, chrms=CHROMNAMES, method=METHOD,
                                                         resolution=RESOLUTION, expected=EXPECTED_MEAN_TAD, exp=STAGE_NAME,
                                                         percentile=PERCENTILE, eps=EPSILON, window_eps=WINDOW_SPOTLIGHT,
                                                     merge_boundaries=MERGE_BOUNDARIES, k=NOISE_SHIFT,
                                                     loc_size=LOC_SIZE, N=NUM_STAGES_TO_MERGE, filtration=FILTRATION,
                                                     bs_thresholds=THRESHOLDS_DATA,
                                                     bs_thresholds_grid=THRESHOLDS_GRID_DATA)
        if not CONSENSUS:
            viz_opt_curves(stats, METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000),
                           EXPERIMENT_PATH, df_conc, RESOLUTION, stage=STAGE_NAME)
        else:
            for STAGE in [x.strip() for x in STAGE_NAME.split(',')]:
                viz_opt_curves(stats[STAGE], METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000),
                               EXPERIMENT_PATH, df_conc, RESOLUTION, stage=STAGE)
        viz_tads(EXPERIMENT_PATH, df_conc, DATASETS, CHROMNAMES, VIZ_TAD_STAGE, RESOLUTION, method=None,
                 is_insulation=True, clusters=False, colors=None, percentile=99.9, vbc=VIZ_BIN_COUNT, consensus=CONSENSUS)
    else:
        opgs, df, df_conc = search_opt_gamma(DATASETS, EXPERIMENT_PATH, method=METHOD,
                                             grid=np.arange(GRID[0], GRID[1], GRID[2]), mis=MAX_INTERTAD, mts=MAX_TAD_SIZE,
                                             start_step=GRID[2], chrms=CHROMNAMES, eps=EPSILON, expected=EXPECTED_MEAN_TAD,
                                             exp=STAGE_NAME, resolution=RESOLUTION, percentile=PERCENTILE)
        viz_opt_curves(df, METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000),
                       EXPERIMENT_PATH, df_conc, RESOLUTION, stage=STAGE_NAME)
        viz_tads(EXPERIMENT_PATH, df_conc, DATASETS, CHROMNAMES, VIZ_TAD_STAGE, RESOLUTION, method=None,
                 is_insulation=False, clusters=False, colors=None, percentile=99.9, vbc=VIZ_BIN_COUNT, consensus=False)

    time_elapsed = time.time() - in_time

    logging.info("RUN_PIPELINE| whole pipeline completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    run_pipeline()
