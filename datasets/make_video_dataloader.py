from datasets import data_manager
from torch.utils.data import DataLoader

from datasets.sampler import RandomIdentitySampler_Video, IdentityBatchSamplerVideo
from datasets.reliable_positive import load_rhp_graph
from dataset_transformer import temporal_transforms as TT, spatial_transforms as ST
from datasets.video_loader import VideoDataset, VideoDatasetInfer

import model.clip


def make_dataloader(cfg):
    dataset_options = {}
    if cfg.DATASETS.NAMES == 'ag_vpreid':
        dataset_options = dict(direction=cfg.DATASETS.AGVPREID_DIRECTION,
                               split_dir=cfg.DATASETS.AGVPREID_SPLIT_DIR)
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR,
                                        **dataset_options)
    if cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM:
        from datasets.dynamic_dfgs import platform_labels_from_cameras
        platform_map = cfg.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM
        if hasattr(dataset, 'camera_to_platform') and tuple(platform_map) != dataset.camera_to_platform:
            raise ValueError('Configured camera-to-platform mapping disagrees with dataset metadata')
        platform_labels_from_cameras([camera for _, _, camera in dataset.train], platform_map)
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

    if cfg.MODEL.CROSS_PLATFORM_SET_WEIGHT:
        mapping = tuple(cfg.MODEL.CROSS_PLATFORM_SET_CAMERA_TO_PLATFORM)
        if hasattr(dataset, 'camera_to_platform') and mapping != tuple(dataset.camera_to_platform):
            raise ValueError('Set-loss camera mapping disagrees with dataset platform metadata')
        cameras = {int(camera) for _, _, camera in dataset.train}
        if any(camera < 0 or camera >= len(mapping) for camera in cameras):
            raise ValueError('Set-loss platform mapping does not cover training cameras')
        if {mapping[camera] for camera in cameras} != {0, 1}:
            raise ValueError('Set-loss training data must contain both platforms')
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
    if sampler_name == 'random_baseline_k4':
        if cfg.DATALOADER.DYNAMIC_DFGS_ENABLED:
            raise ValueError('Random baseline must not enable dynamic DFGS')
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH,
            hard_negative_probability=0.0,
            baseline_replacement_sampling=True)
    if sampler_name == 'softmax_triplet':
        # Reproduction path: preserve the sampler used by the original paper.
        # In particular, identities with fewer than K tracklets draw all K
        # entries with replacement instead of retaining every real tracklet.
        return RandomIdentitySampler_Video(
            train_data, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    if sampler_name == 'reliable_hard_positive':
        graph_path = cfg.DATALOADER.RHP_GRAPH
        if not graph_path:
            raise ValueError('DATALOADER.RHP_GRAPH is required for RHP sampling')
        rhp_graph, _ = load_rhp_graph(graph_path, train_data)
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            reliable_positive_graph=rhp_graph,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH,
            baseline_replacement_sampling=True)
    if sampler_name in ('dfgs', 'mixed_dfgs',
                        'mixed_dfgs_baseline_k4'):
        import json
        graph_path = cfg.DATALOADER.DFGS_GRAPH
        dynamic_enabled = bool(cfg.DATALOADER.DYNAMIC_DFGS_ENABLED)
        if not graph_path and not dynamic_enabled:
            raise ValueError('DATALOADER.DFGS_GRAPH is required for DFGS sampling')
        if graph_path:
            with open(graph_path, 'r') as graph_file:
                raw_graph = json.load(graph_file)
            graph = {int(pid): [int(n) for n in neighbors] for pid, neighbors in raw_graph.items()}
        else:
            # The graph is populated immediately before dynamic Stage-2 epoch 1.
            graph = {}
        return IdentityBatchSamplerVideo(
            train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            neighbor_graph=graph,
            samples_per_pid=cfg.DATALOADER.PID_SAMPLES_PER_EPOCH,
            hard_negative_probability=(cfg.DATALOADER.DFGS_PROBABILITY
                                       if sampler_name in ('mixed_dfgs', 'mixed_dfgs_baseline_k4')
                                       else 1.0),
            baseline_replacement_sampling=sampler_name == 'mixed_dfgs_baseline_k4')
    raise ValueError('Unknown video sampler: {}'.format(cfg.DATALOADER.SAMPLER))
