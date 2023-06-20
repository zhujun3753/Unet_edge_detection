from __future__ import print_function

import argparse
import os
import time, platform

import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from losses import *
from model import DexiNed
from utils import (image_normalization, save_image_batch_to_disk, visualize_result,count_parameters)
from unet import  EncoderDecoder, compute_Image_gradients

import torch.nn as nn


IS_LINUX = True if platform.system() == "Linux" else False

def train_one_epoch(epoch, dataloader, model, criterion_reconstruction, optimizer, device, tb_writer, args=None):
    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)
    # Put model in training mode
    model.train()

    loss_avg = []

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW  torch.Size([8, 3, 352, 352])

        # get the 7 channel we wish to create
        concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
        concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)
        # Forward pass
        encoded, decoded = model(concatenated_smoothed_batch)

        # loss for encoder decoder and add to current loss
        loss = criterion_reconstruction(decoded, concatenated_smoothed_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg.append(loss.item())

        if  (batch_id == len(dataloader) - 1 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('tmp_loss', tmp_loss, epoch)

        if batch_id % 5 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'.format(epoch, batch_id, len(dataloader), loss.item()))

    loss_avg = np.array(loss_avg).mean()
    return loss_avg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EncoderDecoder trainer.')
    parser.add_argument('--is_testing',type=int,
                        default = 0,
                        help='Script in testing mode.')
    # ----------- test -------0--

    # Training settings
    TRAIN_DATA = DATASET_NAMES[0] # BIPED=0, MDBD=6
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']


    parser.add_argument('--input_dir',
                        type=str,
                        default = train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='ecn_mycheckpoints',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type = str,
                        choices = DATASET_NAMES,
                        default = TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--train_list',
                        type = str,
                        default = train_inf['train_list'],
                        help = 'Dataset sample indices list.')
    parser.add_argument('--resume',
                        type = bool,
                        default = False,
                        help = 'use previous trained data')  # Just for test
    
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='34/24_model.pth',# 4 6 7 9 14
                        help='Checkpoint path from which to restore model weights from.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr',
                        default=1e-4,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-8,
                        metavar='WD',
                        help='weight decay (Good 1e-8) in TF1=0') # 1e-8 -> BIRND/MDBD, 0.0 -> BIPED
    parser.add_argument('--batch_size',
                        type=int,
                        default = 16,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        # default = 16,
                        default = 4,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=352,
                        help='Image width for training.') # BIPED 400 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=352,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    # Tensorboard summary writer
    tb_writer = None
    training_dir = os.path.join(args.output_dir, args.train_data)
    os.makedirs(training_dir, exist_ok = True)
    candi_epoch = sorted([int(v) for v in os.listdir(os.path.join(args.output_dir, args.train_data)) if v.isdigit()], reverse=True)
    candi_checkpoint_path = [os.path.join(args.output_dir, args.train_data,str(v), str(v)+'_model.pth') for v in candi_epoch]
    ini_epoch = 0
    for i in range(len(candi_epoch)):
        candi_path = candi_checkpoint_path[i]
        ini_epoch = candi_epoch[i]+1
        if os.path.exists(candi_path):
            checkpoint_path = candi_path
            print("checkpoint_path: ", checkpoint_path)
            break
    # import pdb;pdb.set_trace()
    if args.tensorboard and not args.is_testing:
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir = training_dir)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # Instantiate model and move it to the computing device
    model = EncoderDecoder(input_channels = 7).to(device) #* 17262977

    if not args.is_testing:
        if args.resume:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print('Training restarted from> ',checkpoint_path)
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     arg=args)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size = args.batch_size,
                                      shuffle = True,
                                      num_workers = args.workers)


    criterion_reconstruction = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    # Main training loop
    seed = 1021

    # import pdb;pdb.set_trace()
    for epoch in range(ini_epoch, args.epochs):
        output_dir_epoch = os.path.join(args.output_dir, args.train_data, str(epoch))
        os.makedirs(output_dir_epoch, exist_ok=True)
        avg_loss = train_one_epoch(epoch,
                        dataloader_train,
                        model,
                        criterion_reconstruction,
                        optimizer,
                        device,
                        tb_writer,
                        args = args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss', avg_loss, epoch+1)
        print('Current learning rate> ', optimizer.param_groups[0]['lr'])
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('Encoder Decoder, # of Parameters:')
    print(num_param)
    print('-------------------------------------------------------')

if __name__ == '__main__':
    args = parse_args()
    main(args)
