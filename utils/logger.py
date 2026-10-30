import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train, filename=None, mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
