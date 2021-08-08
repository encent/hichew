import cooler
import glob
import logging
import time
import warnings

from os import listdir
from os.path import basename, join, splitext, isfile, isdir

warnings.filterwarnings("ignore")


def cool_files(path, resolution=5000, chromnames=None):
    """
    Function to load .cool or .mcool files.
    :param path: path to the directory with .cool or .mcool files OR path to the certain coolfile.
    :param resolution: resolution of Hi-C maps in bp (available only for .mcool files)
    :param chromnames: list of chromosomes to be loaded. If None -- all chromosomes will be loaded.
    :return:
    *   nested python dictionary with keys of coolfile names and chromosomes,
    and values of the corresponding contact matrices;
    *   python dictionary with keys of coolfile names and values of the Cooler objects.
    """
    if isfile(path) and (splitext(basename(path))[1] == '.cool' or splitext(basename(path))[1] == '.mcool'):
        files = [path]
    else:
        files = [x for x in glob.glob(join(path, '*.*cool'))]

    if isdir(path):
        if len(set([splitext(x)[1] for x in listdir(path)])) != 1:
            logging.info("LOADER|COOL_FILES| Error: more than one file extension")
            raise Exception('You have files with different extensions in the directory {}. '
                            'Please, leave in the directory only files with the extension .cool OR .mcool'.format(path))

    logging.info("LOADER|COOL_FILES| List of coolfiles of interest: {}".format(str(files)))

    labels = [splitext(basename(x))[0] for x in files]
    matrices = {x: {} for x in labels}
    coolers = {x: None for x in labels}

    logging.info("LOADER|COOL_FILES| Start loading coolfiles...")
    in_time = time.time()

    for label, file in list(zip(labels, files)):
        if splitext(basename(file))[1] == '.mcool':
            try:
                c = cooler.Cooler(file + '::/resolutions/{}'.format(resolution))
            except Exception as e:
                c = cooler.Cooler(file + '::/resolution/{}'.format(resolution))
        elif splitext(basename(file))[1] == '.cool':
            c = cooler.Cooler(file)
        else:
            logging.info("LOADER|COOL_FILES| Error: file {} is not a coolfile!".format(basename(file)))
            raise Exception('Not a coolfile!')

        coolers[label] = c

        if chromnames:
            chrms = chromnames
        else:
            chrms = c.chromnames

        for ch in chrms:
            mtx = c.matrix(balance=True).fetch(ch)
            matrices[label][ch] = mtx.copy()

    time_elapsed = time.time() - in_time
    logging.info("LOADER|COOL_FILES| Loading completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return matrices, coolers
