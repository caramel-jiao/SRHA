"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torch.nn.functional as F
import numpy as np
import random

### set logging
logging.getLogger().setLevel(logging.INFO)

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decoder_lr', type=float, default=None)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture. We have DeepR50V3PlusD (backbone: ResNet50) \
                    and DeepR101NV3PlusD (backbone: ResNet101).')
parser.add_argument('--style_elimination', action='store_true', default=True)

parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--combine_all', action='store_true', default=False,
                    help='combine both train, val, and test sets of the source data')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling, this one should not exceed 760 for synthia')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--early_stop_iter', type=int, default=10000000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=12,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--gtav_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--synthia_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

parser.add_argument('--eval_epoch', type=int, default=1,
                    help='eval interval')

## domain-neutral knowledge
parser.add_argument('--teacher', action='store_true', default=True,
                    help=' teacher: imgnet model')
## concentration parameter for Dir, 2^{-6}
parser.add_argument('--concentration_coeff', type=float, default=0.0156,
                    help='coefficient for concentration')
## SRM
parser.add_argument('--srm', action='store_true', default=True,
                    help='use style rearrangement module')
## global alignment
parser.add_argument('--ga_layer', nargs='*', type=str, default=['layer1', 'layer2', 'layer3', 'layer4'],
                    help='a list of layers for global alignment : layer 0,1,2,3,4')
parser.add_argument('--ga_weight', nargs='*', type=float, default=[0.4, 0.6, 0.8, 1.0],
                    help='global alignment weight')
## regional alignment
parser.add_argument('--ra_layer', nargs='*', type=str, default=['layer1', 'layer2', 'layer3', 'layer4'],
                    help='a list of layers for regional alignment : layer 0,1,2,3,4')
parser.add_argument('--ra_weight', nargs='*', type=float, default=[0.4, 0.6, 0.8, 1.0],
                    help='regional alignment weight')
## local alignment
parser.add_argument('--la_layer', nargs='*', type=str, default=['layer1', 'layer2', 'layer3', 'layer4'],
                    help='a list of layers for retrospection loss : layer 0,1,2,3,4')
parser.add_argument('--la_weight', nargs='*', type=float, default=[0.4, 0.6, 0.8, 1.0],
                    help='weight for each layer feature of retrospection layer')
## prediction consistency
parser.add_argument('--pc_weight', type=float, default=10.0,
                    help='weight for prediction consistency')

## EMA alpha
parser.add_argument('--alpha', type=float, default=1,
                    help='factor of ema teacher update')
## freeze layers
parser.add_argument('--freeze_layer0', action='store_true', default=False,
                    help='freeze layer0 or not')
parser.add_argument('--freeze_layer1', action='store_true', default=False,
                    help='freeze layer1 or not')
parser.add_argument('--freeze_layer2', action='store_true', default=False,
                    help='freeze layer2 or not')
parser.add_argument('--freeze_layer3', action='store_true', default=False,
                    help='freeze layer3 or not')
parser.add_argument('--freeze_layer4', action='store_true', default=False,
                    help='freeze layer4 or not')


args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1


if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
# args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                    #  init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)


def prepare_targets(gt_tensor, num_classes=19):
    ret_target = []
    for label_tensor in gt_tensor:
        label_list = []
        mask_list = []
        for cls_idx in range(num_classes):
            label_list.append(cls_idx)
            mask_list.append(label_tensor == cls_idx)
        labels = torch.LongTensor(label_list).to(gt_tensor.device)
        masks = torch.stack(mask_list, dim=0)
        target = {'labels': labels, 'masks': masks}
        ret_target.append(target)
    return ret_target


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    
    net = network.get_net(args)
    optim, scheduler = optimizer.get_encoder_decoder_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    
    epoch = 0
    i = 0
    best_mean_iu = 0
    best_epoch = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    if args.local_rank == 0:
        msg_args = ''
        args_dict = vars(args)
        for k, v in args_dict.items():
            msg_args = msg_args + str(k) + ' : ' + str(v) + ', '
        logging.info(msg_args)

    if args.teacher:
        teacher_model = network.get_net(args)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        teacher_model = network.warp_network_in_dataparallel(teacher_model, args.local_rank)
    else:
        teacher_model = None

    train_max_iter = min(args.max_iter, args.early_stop_iter)

    while i < train_max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        print("#### iteration", i)
        torch.cuda.empty_cache()

        i = train(train_loader, net, optim, epoch, writer, scheduler, 
                  train_max_iter, teacher_model, criterion, criterion_aux,
                  args.alpha)
        train_loader.sampler.set_epoch(epoch + 1)

        if (epoch+1) % args.eval_epoch == 0 or i >= train_max_iter:
            # torch.cuda.empty_cache()
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

            for dataset, val_loader in extra_val_loaders.items():
                print("Extra validating... This won't save pth file")
                mean_iu = validate(val_loader, dataset, net, criterion_val, optim, scheduler, 
                                   epoch, writer, i, save_pth=False)

                if args.local_rank == 0:
                    if mean_iu > best_mean_iu:
                        best_mean_iu = mean_iu
                        best_epoch = epoch
                    
                    msg = 'Best Epoch:{}, Best mIoU:{:.5f}'.format(best_epoch, best_mean_iu)
                    logging.info(msg)
                break

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()

        epoch += 1

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        miou = validate(val_loader, dataset, net, criterion_val, optim, scheduler, 
                        epoch, writer, i, save_pth=False)

def train(train_loader, net, optim, curr_epoch, writer, scheduler, 
          max_iter, teacher_model, criterion, criterion_aux,
          alpha):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    if teacher_model is not None:
        teacher_model.train()
        network.freeze_dropout_ema(teacher_model)

    train_total_loss = AverageMeter()
    la_loss_meter = AverageMeter()
    ga_loss_meter = AverageMeter()
    ra_loss_meter = AverageMeter()
    pc_loss_meter = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        
        if curr_iter == 0 and alpha < 1:
            network.init_ema_weights(net, teacher_model)
        if curr_iter > 0 and alpha < 1:
            network.update_ema(net, teacher_model, curr_iter, alpha)

        inputs, gts, _, aux_gts = data
        
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W
            
        feat_layer_list = np.unique(
            [ga for idx, ga in enumerate(args.ga_layer) if args.ga_weight[idx] > 0] + \
            [ra for idx, ra in enumerate(args.ra_layer) if args.ra_weight[idx] > 0] + \
            [la for idx, la in enumerate(args.la_layer) if args.la_weight[idx] > 0]
        ).tolist()
            
        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            input, gt, aux_gt = ingredients

            start_ts = time.time()

            input, gt, aux_gt = input.cuda(), gt.cuda(), aux_gt.cuda()
            
            gt = torch.cat((gt, gt), dim=0)
            aux_gt = torch.cat((aux_gt, aux_gt), dim=0)
            if teacher_model is not None:
                with torch.no_grad():
                    imgnet_out = teacher_model(input, seg_gt=gt, out_prob=True, 
                            return_style_features=feat_layer_list)

            optim.zero_grad()
            
            outputs = net(input, seg_gt=gt, out_prob=True, return_style_features=feat_layer_list)

            main_out = outputs['main_out']
            
            # get target for m2f loss
            m2f_target = prepare_targets(gt)
            m2f_outputs = outputs['m2f_outputs']

            criterion = net.module.criterion
            main_loss = criterion(m2f_outputs, m2f_target)
            for k in list(main_loss.keys()):
                if k in criterion.weight_dict:
                    main_loss[k] *= criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    main_loss.pop(k)
            # decoder has 9 layers in total, cal all layers' loss
            total_loss = sum([ml for ml in main_loss.values()])
        
            # # bad
            # main_loss = criterion(main_out, gt)
            # total_loss = main_loss
            
            total_ga_loss = 0.
            for layer, l_w in zip(args.ga_layer, args.ga_weight):
                if l_w > 0:
                    tea_ori_feat = imgnet_out['features'][layer][: B]
                    _f_imgnet = torch.cat((tea_ori_feat, tea_ori_feat), dim=0).detach()
                    dgf_style = F.adaptive_avg_pool2d(outputs['features'][layer], 1)
                    dgf_imgnet = F.adaptive_avg_pool2d(_f_imgnet, 1).detach()
                    ga_loss = F.mse_loss(dgf_style, dgf_imgnet)
                    total_ga_loss = total_ga_loss + l_w * ga_loss
            if isinstance(total_ga_loss, torch.Tensor):
                total_loss = total_loss + total_ga_loss
                ga_loss_meter.update(total_ga_loss.item(), C)
                
            total_ra_loss = 0.
            for layer, l_w in zip(args.ra_layer, args.ra_weight):
                if l_w > 0:
                    tea_ori_feat = imgnet_out['features'][layer][: B]
                    _f_imgnet = torch.cat((tea_ori_feat, tea_ori_feat), dim=0).detach()
                    _f_style = outputs['features'][layer]
                    deep_gt = gt.unsqueeze(1).float()
                    deep_gt = F.interpolate(deep_gt, size=_f_style.shape[2:], mode='nearest')
                    deep_gt = deep_gt.squeeze(1).long()
                    ra_loss = loss.cal_semantic_regional_align(deep_gt, _f_imgnet, _f_style)
                    total_ra_loss = total_ra_loss + l_w * ra_loss
            if isinstance(total_ra_loss, torch.Tensor):
                total_loss = total_loss + total_ra_loss
                ra_loss_meter.update(total_ra_loss.item(), C)
                
            total_la_loss = 0.
            for layer, l_w in zip(args.la_layer, args.la_weight):
                if l_w > 0:
                    f_style = outputs['features']
                    f_imgnet = imgnet_out['features']
                    tea_ori_feat = f_imgnet[layer][:B]
                    _f_imgnet = torch.cat((tea_ori_feat, tea_ori_feat), dim=0).detach()
                    _f_style = f_style[layer]
                    la_loss = loss.cal_local_align(_f_imgnet, _f_style)
                    total_la_loss = total_la_loss + l_w * la_loss
            if isinstance(total_la_loss, torch.Tensor):
                total_loss = total_loss + total_la_loss
                la_loss_meter.update(total_la_loss.item(), C)
                
            if args.pc_weight:
                outputs_sm = F.softmax(main_out, dim=1)
                im_prob = outputs_sm[:B] 
                aug_prob = outputs_sm[B:] 

                aug_prob = aug_prob.permute(0,2,3,1).reshape(-1, main_out.shape[1])
                im_prob = im_prob.permute(0,2,3,1).reshape(-1, main_out.shape[1])
                
                pc_loss = args.pc_weight * loss.calc_js_div_loss(im_prob, aug_prob)
                total_loss = total_loss + pc_loss

                pc_loss_meter.update(pc_loss.item(), batch_pixel_size)
                
            
            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss
                
            if args.local_rank == 0:
                if i % 30 == 29:

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], ' \
                        .format(curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg)
                           
                    if ga_loss_meter.count > 0:
                        msg += '[global alignment loss {:0.6f}], '.format(ga_loss_meter.avg)
                    if ra_loss_meter.count > 0:
                        msg += '[regional alignment loss {:0.6f}], '.format(ra_loss_meter.avg)
                    if la_loss_meter.count > 0:
                        msg += '[local alignment loss {:0.6f}], '.format(la_loss_meter.avg)
                    if pc_loss_meter.count > 0:
                        msg += '[prediction consistency loss {:0.6f}], '.format(pc_loss_meter.avg)
                        
                    msg += '[enc_lr {:0.6f}], [dec_lr {:0.6f}], [time {:0.4f}]'.format(
                        optim.param_groups[0]['lr'], optim.param_groups[-1]['lr'], time_meter.avg)

                    logging.info(msg)
                    
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 20 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, 
             curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)
        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps, invalid
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)
    else:
        mean_iu = 0

    return mean_iu


if __name__ == '__main__':
    main()
