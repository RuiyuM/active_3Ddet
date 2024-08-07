import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import wandb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="0"


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/active-kitti_models/pv_rcnn_active_montecarlo_from_scratch.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training') 
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='select-100', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=1, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--eval_backbone', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()
    

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    common_utils.set_random_seed(1024)
    args.extra_tag = 'select-{}'.format(cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS)


    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, cur_epoch_id, dist_test=False):
    # eval_utils.vis_one_epoch(
    #     cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
    #     result_dir=eval_output_dir, save_to_file=args.save_to_file
    # )
    ckpt_record_file = eval_output_dir / ('vis_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(
            log_dir=str(eval_output_dir / ('tensorboard_vis_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
        # wandb.init(project=cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + '_vis', entity='user')
        wandb.init(project=cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + '_vis_select-{}'.format(
            cfg.ACTIVE_TRAIN.SELECT_NUMS), entity="data-efficient-lab")
        run_name_elements = [cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0]] + [
            'backbone' if args.eval_backbone else cfg.TAG] + [cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS] + [
                                cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL] + [
                                cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS] + [cfg.ACTIVE_TRAIN.SELECT_NUMS]
        run_name_elements = '_'.join([str(i) for i in run_name_elements])
        wandb.run.name = run_name_elements
        wandb.config.update(args)
        wandb.config.update(cfg)

    # start evaluation
    cur_result_dir = eval_output_dir
    tb_dict = eval_utils.vis_one_epoch(
        cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
        result_dir=cur_result_dir, save_to_file=args.save_to_file
    )
    # cur_epoch_id = str(int(cur_epoch_id) - 40)
    # active_label_files = glob.glob(str(args.set_cfgs / 'active_label' / 'selected_frames_epoch_*.pkl'))

    # active_label_files = [active_label_file]
    num_bbox = 0
    for i in range(4):
        z = i + 1
        new_epoch_id = str(int(cur_epoch_id) - z *40)
        active_label_file = os.path.join(args.set_cfgs, 'active_label',
                                         f'selected_frames_epoch_{new_epoch_id}_rank_0.pkl')
        with open(active_label_file, 'rb') as f:
            records = pickle.load(f)
            num_bbox += sum([sum(i.values()) for i in records['selected_bbox']])

    num_bbox = int(num_bbox)
    if cfg.LOCAL_RANK == 0:
        for key, val in tb_dict.items():
            tb_log.add_scalar(key, val, num_bbox)
            wandb.log({key: val}, step=num_bbox)

    # record this epoch which has been evaluated
    with open(ckpt_record_file, 'a') as f:
        print('%s' % cur_epoch_id, file=f)
    logger.info('Epoch %s has been visualised' % cur_epoch_id)


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if "select" in epoch_id: # active training epochs
            epoch_id = epoch_id.split('_')[0]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('vis_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_vis_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
        # wandb.init(project=cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + '_vis', entity='user')
        wandb.init(project=cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + '_vis_select-{}'.format(cfg.ACTIVE_TRAIN.SELECT_NUMS), entity="data-efficient-lab")
        run_name_elements = [cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0]] + ['backbone' if args.eval_backbone else cfg.TAG] + [cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS] + [cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL] + [cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS] + [cfg.ACTIVE_TRAIN.SELECT_NUMS]
        run_name_elements = '_'.join([str(i) for i in run_name_elements])
        wandb.run.name = run_name_elements
        wandb.config.update(args)
        wandb.config.update(cfg)
    
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        # model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        # model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.vis_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        active_label_files = glob.glob(str(eval_output_dir.parents[2] / 'active_label' / 'selected_frames_epoch_*.pkl'))
        active_label_files = [file for file in active_label_files if int(file.split('.')[0].split('_')[-3]) <= int(cur_epoch_id)]
        num_bbox = 0
        for file in active_label_files:
            with open(file, 'rb') as f:
                records = pickle.load(f)
                num_bbox += sum([sum(i.values()) for i in records['selected_bbox']])
        num_bbox = int(num_bbox)
        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, num_bbox)
                wandb.log({key: val}, step=num_bbox)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been visualised' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.eval_backbone:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'backbone' / args.extra_tag
        output_dir.mkdir(parents=True, exist_ok=True)
       
    else:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_vis_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    if args.eval_backbone:
        logger.info('start visualizing backbones for {}...'.format(cfg.EXP_GROUP_PATH))
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'
    args.set_cfgs = output_dir
    args.ckpt_dir = ckpt_dir

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # no need model to visualise
    model = None
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
