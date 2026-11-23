import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.frame_head_supervision import FrameHeadSupervision
import einops

from utils.test_video_reid import test, _eval_format_logger
from datasets.dynamic_dfgs import refresh_dynamic_dfgs_graph, should_refresh_graph
from utils.reproducibility import reset_training_seed


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             query_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             local_rank,
             optimizer_stage1):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    final_checkpoint_only = cfg.SOLVER.STAGE2.FINAL_CHECKPOINT_ONLY
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    aggregation_enabled = cfg.MODEL.FRAME_AGGREGATION != 'mean'
    diversity_weight = cfg.MODEL.FRAME_AGGREGATION_DIVERSITY_WEIGHT
    head_loss_weight = cfg.MODEL.FRAME_AGGREGATION_HEAD_LOSS_WEIGHT
    head_criterion = FrameHeadSupervision(cfg) if head_loss_weight else None
    if aggregation_enabled:
        logger.info('Frame aggregation: training/retrieval=%s heads=%d groups=%d diversity-weight=%.4f; '
                    'dynamic DFGS center/reliability readout=%s',
                    cfg.MODEL.FRAME_AGGREGATION, cfg.MODEL.FRAME_AGGREGATION_HEADS,
                    cfg.MODEL.FRAME_AGGREGATION_GROUPS, diversity_weight, cfg.DATALOADER.DYNAMIC_DFGS_READOUT)
        if cfg.MODEL.FRAME_AGGREGATION_GROUPS > 1:
            logger.info('Feature-group frame scores: weights=[batch,heads,groups,frames]; '
                        'softmax is over frames, statistics average all heads/groups; '
                        'feature groups are channel partitions, not predefined body parts')
        if cfg.MODEL.FRAME_AGGREGATION == 'single_shared':
            logger.info('Single shared attention: one learned frame distribution repeated into two fusion slots; '
                        'head-overlap=1 is expected, diversity regularization is disabled')
        if cfg.MODEL.FRAME_AGGREGATION.startswith('conditional'):
            logger.info('Conditional controls: mode=%s head-loss-weight=%.4f suppression=%.3f '
                        'gate-init=%.3f; centered bias-free residuals, live shared head classifier',
                        cfg.MODEL.FRAME_AGGREGATION, head_loss_weight,
                        cfg.MODEL.FRAME_AGGREGATION_SUPPRESSION,
                        cfg.MODEL.FRAME_AGGREGATION_GATE_INIT)
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    diversity_meter = AverageMeter()
    entropy_meter = AverageMeter()
    max_weight_meter = AverageMeter()
    head_id_meter, head_triplet_meter = AverageMeter(), AverageMeter()
    gate_meter, primary_residual_meter, supplement_residual_meter = AverageMeter(), AverageMeter(), AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # torch.save(model.state_dict(),
    #            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'initial_weight.pth'))
    # assert 1 < 0
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    print("collect all text features")
    with torch.no_grad():
        # model.without_prompt_(True)
        # model.without_dat_(True)
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
                # print(text_feature.shape)
                # assert 1 < 0
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
    # model.without_prompt_(False)
    # model.without_dat_(False)

    # model.change_stage(stage1=False)
    print(text_features[:5])
    print("text features has been collected, begin to train!")
    best_rank_1 = 0.0
    best_mAP =0.0
    if cfg.SOLVER.STAGE2.RESET_SEED:
        reset_training_seed(cfg.SOLVER.SEED)
        logger.info("Reset Stage-2 training seed to %d; cuDNN benchmark disabled",
                    cfg.SOLVER.SEED)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        diversity_meter.reset()
        entropy_meter.reset()
        max_weight_meter.reset()
        for meter in (head_id_meter, head_triplet_meter, gate_meter,
                      primary_residual_meter, supplement_residual_meter):
            meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        if (cfg.DATALOADER.DYNAMIC_DFGS_ENABLED and
                should_refresh_graph(
                    epoch, cfg.DATALOADER.DYNAMIC_DFGS_UPDATE_PERIOD)):
            refresh_dynamic_dfgs_graph(
                cfg, model, train_loader_stage2.sampler, epoch, logger)

        model.train()
        for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage2):
            # print(model.prompt_learner.text_shared_prompts[0])
            # TODO
            target_view = target_cam
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            if cfg.MODEL.PBP_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.PBP_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                if aggregation_enabled:
                    score, feat, image_features, aggregation_aux = model(
                        x=img, label=target, cam_label=target_cam,
                        view_label=target_view, return_aggregation_aux=True)
                else:
                    score, feat, image_features = model(
                        x=img, label=target, cam_label=target_cam,
                        view_label=target_view)
                text_feature = model(label=target, get_text=True)
                loss_i2t = xent(image_features, text_feature, target, target)
                loss_t2i = xent(text_feature, image_features, target, target)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits, loss_t2i, loss_i2t)
                if aggregation_enabled and diversity_weight:
                    loss = loss + diversity_weight * aggregation_aux['diversity_loss'].mean()
                if head_criterion is not None:
                    head_loss, head_details = head_criterion(aggregation_aux, target)
                    loss = loss + head_loss_weight * head_loss

                # 更新样本中心


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            if aggregation_enabled:
                weights = aggregation_aux['weights'].detach().float()
                diversity_meter.update(float(aggregation_aux['diversity_loss'].detach().mean()), img.shape[0])
                entropy_meter.update(float(-(weights * weights.clamp_min(1e-12).log()).sum(-1).mean()), img.shape[0])
                max_weight_meter.update(float(weights.max(-1).values.mean()), img.shape[0])
                if head_criterion is not None:
                    head_id_meter.update(float(head_details['head_id_loss'].detach()), img.shape[0])
                    head_triplet_meter.update(float(head_details['head_triplet_loss'].detach()), img.shape[0])
                if 'supplement_gate' in aggregation_aux:
                    gate_meter.update(float(aggregation_aux['supplement_gate'].detach().mean()), img.shape[0])
                    primary_residual_meter.update(float(aggregation_aux['primary_residual_relative_norm'].detach().mean()), img.shape[0])
                    supplement_residual_meter.update(float(aggregation_aux['supplement_residual_relative_norm'].detach().mean()), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        if aggregation_enabled:
            logger.info('Frame aggregation epoch means: cosine-head-overlap=%.6f entropy=%.6f '
                        'max-weight=%.6f diversity-weight=%.4f',
                        diversity_meter.avg, entropy_meter.avg, max_weight_meter.avg,
                        diversity_weight)
            if cfg.MODEL.FRAME_AGGREGATION.startswith('conditional'):
                logger.info('Conditional epoch means: head-ID=%.6f head-triplet=%.6f '
                            'head-loss-weight=%.4f supplement-gate=%.6f '
                            'primary-residual-ratio=%.6f supplement-residual-ratio=%.6f',
                            head_id_meter.avg, head_triplet_meter.avg, head_loss_weight,
                            gate_meter.avg, primary_residual_meter.avg, supplement_residual_meter.avg)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))
        if cfg.DATALOADER.DYNAMIC_DFGS_SOFT_NEGATIVES:
            stats = train_loader_stage2.sampler.soft_sampling_stats
            logger.info("Soft negative sampling epoch %d: graph_batches=%d, low_neighbor_batches=%d, random_batches=%d",
                        epoch, stats["graph_batches"], stats["low_neighbor_batches"],
                        stats["random_batches"])

        if epoch % checkpoint_period == 0 or (final_checkpoint_only and epoch == epochs):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if ((not final_checkpoint_only and epoch % eval_period == 0) or
                (final_checkpoint_only and epoch == epochs)):
            if cfg.MODEL.DIST_TRAIN:
                assert 1 < 0
            else:
                use_gpu = True
                cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
                ptr = "mAP: {:.2%}".format(mAP)
                if not final_checkpoint_only and cmc[0] > best_rank_1:
                    best_rank_1 = cmc[0]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_rank1_best.pth'))
                if not final_checkpoint_only and mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_mAP_best.pth'))
                for r in ranks:
                    ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                logger.info(ptr)

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
