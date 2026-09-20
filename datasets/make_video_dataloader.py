from datasets import data_manager
from torch.utils.data import DataLoader

from datasets.sampler import RandomIdentitySampler_Video, IdentityBatchSamplerVideo
from dataset_transformer import temporal_transforms as TT, spatial_transforms as ST
from datasets.video_loader import VideoDataset, VideoDatasetInfer

import model.clip


def make_dataloader(cfg):
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    spatial_transform_train_stage2 = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.RandomHorizontalFlip(cfg.INPUT.PROB),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
        ST.RandomErasing(probability=cfg.INPUT.RE_PROB)])

    spatial_transform_train_stage1 = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])

    spatial_transform_test = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)])

    temporal_transform_train = TT.TemporalRestrictedCrop(size=cfg.DATALOADER.SEQ_LEN)
    temporal_transform_test= TT.TemporalRestrictedBeginCrop(size=cfg.DATALOADER.SEQ_LEN)

    train_loader_stage1 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_train_stage2,
            temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler_Video(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    stage2_sampler = _make_stage2_sampler(cfg, dataset.train)
    train_loader_stage2 = DataLoader(
        VideoDataset(
            dataset.train,
            spatial_transform=spatial_transform_train_stage2,
            temporal_transform=temporal_transform_train),
        sampler=stage2_sampler,
        batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    queryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    galleryloader_sampled_frames = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    queryloader_all_frames = DataLoader(
        VideoDatasetInfer(
            dataset.query, spatial_transform=spatial_transform_test, seq_len=cfg.DATALOADER.SEQ_LEN),
        batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True, drop_last=False)

    galleryloader_all_frames = DataLoader(
        VideoDatasetInfer(dataset.gallery, spatial_transform=spatial_transform_test, seq_len=cfg.DATALOADER.SEQ_LEN),
        batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True, drop_last=False)

    num_classes = dataset.num_train_pids
    num_query = dataset.num_query_pids
    camera_num = dataset.num_camera

    return (train_loader_stage2,
            train_loader_stage1,
            queryloader_sampled_frames,
            galleryloader_sampled_frames,
            queryloader_all_frames,
            galleryloader_all_frames,
            num_classes,
            num_query,
            camera_num)


def _make_stage2_sampler(cfg, train_data):
    sampler_name = str(cfg.DATALOADER.SAMPLER).lower()
    if sampler_name == 'softmax_triplet':
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH)
    if sampler_name == 'cross_camera':
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE, cross_camera=True,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH)
    if sampler_name in ('dfgs', 'dfgs_cross_camera', 'mixed_dfgs',
                        'mixed_dfgs_baseline_k4'):
        import json
        graph_path = cfg.DATALOADER.DFGS_GRAPH
        if not graph_path:
            raise ValueError('DATALOADER.DFGS_GRAPH is required for DFGS sampling')
        with open(graph_path, 'r') as graph_file:
            raw_graph = json.load(graph_file)
        graph = {int(pid): [int(n) for n in neighbors] for pid, neighbors in raw_graph.items()}
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            cross_camera=sampler_name == 'dfgs_cross_camera',
            neighbor_graph=graph,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH,
            hard_negative_probability=(cfg.DATALOADER.DFGS_PROBABILITY
                                       if sampler_name in ('mixed_dfgs', 'mixed_dfgs_baseline_k4')
                                       else 1.0),
            baseline_replacement_sampling=sampler_name == 'mixed_dfgs_baseline_k4')
    raise ValueError('Unknown video sampler: {}'.format(cfg.DATALOADER.SAMPLER))
