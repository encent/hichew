import argparse
import datetime
import logging
import sys
import time
import warnings
from os import makedirs
from os.path import abspath, normpath, join, isdir, splitext

import numpy as np
from api import download_files, load_cool_files, search_opt_gamma, viz_opt_curves, viz_tads, search_opt_window

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
                        help='Name of coolfile (with or without extension) corresponding to the stage of development '
                             'by which TADs optimal segmentation search will be run.')
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

    args = parser.parse_args()

    INPUT_TYPE = args.input_type
    INPUT_PATH = abspath(normpath(args.input_path)) if INPUT_TYPE != 'url' else args.input_path

    EXPERIMENT_NAME = args.experiment if args.experiment != '' else "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    EXPERIMENT_PATH = join('../data/experiments', EXPERIMENT_NAME)
    if not isdir(EXPERIMENT_PATH): makedirs(EXPERIMENT_PATH)

    EPSILON = args.epsilon
    STAGE_NAME = splitext(args.stage)[0]
    RESOLUTION = args.resolution
    CHROMNAMES = args.chromnames; CHROMNAMES = [x.strip() for x in CHROMNAMES.split(',')]
    METHOD = args.method
    GRID = args.grid; GRID = [float(x.strip()) for x in GRID.split(',')]
    EXPECTED_MEAN_TAD = args.expected_mean_tad
    MAX_INTERTAD = args.max_intertad_size
    MAX_TAD_SIZE = args.max_tad_size
    PERCENTILE = args.percentile
    VIZ_BIN_COUNT = args.viz_bin_count

    if METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) > 2 and \
            len(np.arange(GRID[0], GRID[1], GRID[2])) < 10:
        WINDOW_SPOTLIGHT = 2
    elif METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) >= 10:
        WINDOW_SPOTLIGHT = 5
    elif METHOD == 'insulation' and len(np.arange(GRID[0], GRID[1], GRID[2])) <= 2:
        raise Exception('Your grid is too small for "insulation" method!')
    elif METHOD != 'insulation':
        WINDOW_SPOTLIGHT = None

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=join(abspath(normpath(EXPERIMENT_PATH)), 'segmentation.log'), filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("RUN_PIPELINE| running pipeline...")
    logging.info("RUN_PIPELINE| input command line params:\n"
                 "input_type: {}\ninput_path: {}\nexperiment_name: {}\n"
                 "experiment_path: {}\nepsilon: {}\nstage_name: {}\nresolution: {}\n"
                 "chromnames: {}\nmethod: {}\ngrid: {}\n"
                 "expected_mean_tad: {}\nmax_intertad: {}\nmax_tad_size: {}\n"
                 "percentile: {}\nviz_bin_count: {}\nwindow_spotlight: {}".format(INPUT_TYPE, INPUT_PATH,
                                           EXPERIMENT_NAME, EXPERIMENT_PATH, EPSILON, STAGE_NAME, RESOLUTION,
                                           str(CHROMNAMES), METHOD, str(GRID), EXPECTED_MEAN_TAD, MAX_INTERTAD,
                                           MAX_TAD_SIZE, PERCENTILE, VIZ_BIN_COUNT, WINDOW_SPOTLIGHT))

    in_time = time.time()

    COOLFILES_PATH = download_files(INPUT_TYPE, INPUT_PATH)
    DATASETS, COOL_SETS = load_cool_files(COOLFILES_PATH, CHROMNAMES, [STAGE_NAME])
    if METHOD == 'insulation':
        df, df_conc, stats, opws = search_opt_window(DATASETS, COOL_SETS, EXPERIMENT_PATH,
                                                     grid=np.arange(GRID[0], GRID[1], GRID[2]) * RESOLUTION, mis=MAX_INTERTAD,
                                                     mts=MAX_TAD_SIZE, chrms=CHROMNAMES, method=METHOD,
                                                     resolution=RESOLUTION, expected=EXPECTED_MEAN_TAD, exp=STAGE_NAME,
                                                     percentile=PERCENTILE, eps=EPSILON, window_eps=WINDOW_SPOTLIGHT)
        viz_opt_curves(stats, METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000),
                       EXPERIMENT_PATH)
        viz_tads(EXPERIMENT_PATH, df_conc, DATASETS, CHROMNAMES, STAGE_NAME, RESOLUTION, method=None, is_insulation=True, clusters=False,
                 colors=None, percentile=99.9, vbc=VIZ_BIN_COUNT)
    else:
        opgs, df, df_conc = search_opt_gamma(DATASETS, EXPERIMENT_PATH, method=METHOD,
                                             grid=np.arange(GRID[0], GRID[1], GRID[2]), mis=MAX_INTERTAD, mts=MAX_TAD_SIZE,
                                             start_step=GRID[2], chrms=CHROMNAMES, eps=EPSILON, expected=EXPECTED_MEAN_TAD,
                                             exp=STAGE_NAME, resolution=RESOLUTION, percentile=PERCENTILE)
        viz_opt_curves(df, METHOD, CHROMNAMES, EXPECTED_MEAN_TAD / RESOLUTION, int(EXPECTED_MEAN_TAD / 1000), EXPERIMENT_PATH)
        viz_tads(EXPERIMENT_PATH, df_conc, DATASETS, CHROMNAMES, STAGE_NAME, RESOLUTION, method=None, is_insulation=False, clusters=False, colors=None, percentile=99.9, vbc=VIZ_BIN_COUNT)

    time_elapsed = time.time() - in_time

    logging.info("RUN_PIPELINE| whole pipeline completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    run_pipeline()
