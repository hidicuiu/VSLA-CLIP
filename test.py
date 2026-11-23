import os
from config import cfg
import argparse
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import test, EVALUATION_PROTOCOL
from utils.logger import setup_logger


def evaluation_configurations(config):
    """Resolve standalone directions without changing training configuration."""
    if config.DATASETS.NAMES != 'ag_vpreid':
        return [(None, config)]
    directions = list(config.TEST.AGVPREID_DIRECTIONS)
    if not directions:
        directions = [config.DATASETS.AGVPREID_DIRECTION]
    if len(set(directions)) != len(directions) or any(
            direction not in ('g2a', 'a2g') for direction in directions):
        raise ValueError('TEST.AGVPREID_DIRECTIONS must contain unique g2a/a2g values')
    configurations = []
    for direction in directions:
        current = config.clone()
        current.defrost()
        if direction != config.DATASETS.AGVPREID_DIRECTION:
            # Let the reader discover the real opposite-direction directory.
            current.DATASETS.AGVPREID_SPLIT_DIR = ''
        current.DATASETS.AGVPREID_DIRECTION = direction
        current.freeze()
        configurations.append((direction, current))
    return configurations


def run_evaluation(config, config_file='', arguments=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID
    model = None
    results = {}
    for direction, current in evaluation_configurations(config):
        is_ag = direction is not None
        # Append AG evaluations so rerunning never truncates earlier results.
        logger = setup_logger(
            'transreid.eval.{}'.format(direction) if is_ag else 'transreid',
            current.OUTPUT_DIR, if_train=False,
            filename='test_log_{}.txt'.format(direction) if is_ag else None,
            mode='a' if is_ag else 'w')
        handlers = list(logger.handlers)
        try:
            logger.info(arguments)
            logger.info('Evaluation direction: %s; checkpoint: %s',
                        direction.upper() if is_ag else current.DATASETS.NAMES,
                        current.TEST.WEIGHT)
            logger.info('Evaluation protocol: %s', EVALUATION_PROTOCOL)
            if config_file:
                logger.info('Loaded configuration file %s', config_file)
                with open(config_file, 'r') as handle:
                    logger.info('\n' + handle.read())
            logger.info('Running with config:\n%s', current)
            loaders = make_dataloader(current)
            if model is None:
                model = make_model(current, num_class=loaders[6],
                                   camera_num=loaders[8], view_num=0)
                model.load_param(current.TEST.WEIGHT)
            cmc, mAP, ranks = test(model, loaders[2], loaders[3], True, current)
            ptr = 'mAP: {:.2%}'.format(mAP)
            for rank in ranks:
                ptr += ' | R-{:<3}: {:.2%}'.format(rank, cmc[rank - 1])
            logger.info(ptr)
            results[direction or current.DATASETS.NAMES] = {
                'mAP': float(mAP), 'rank1': float(cmc[0])}
            del loaders
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    run_evaluation(cfg, args.config_file, args)
