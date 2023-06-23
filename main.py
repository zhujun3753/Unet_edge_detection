
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
from unet import UNet
from unet import  EncoderDecoder, compute_Image_gradients
from evaluate_model import *
import wandb

import torch.nn as nn
import torch.nn.functional as F

IS_LINUX = True if platform.system()=="Linux" else False

def train_one_epoch(epoch, dataloader, model, ecn_model, criterion, criterion_reconstruction, optimizer, ecn_optimiser,\
                     device, log_interval_vis, tb_writer, args = None):
    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)
    # Put model in training mode
    model.train()
    # l_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1]  # for bdcn ori loss
    # before [0.6,0.6,1.1,1.1,0.4,0.4,1.3] [0.4,0.4,1.1,1.1,0.6,0.6,1.3],[0.4,0.4,1.1,1.1,0.8,0.8,1.3]
    l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3] # New BDCN  loss
    # l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
    #             [0.1, 1.], [0.1, 1.], [0.1, 1.],
    #             [0.01, 4.]]  # for cats loss
    unet_loss_avg = []
    reconstruction_loss_avg = []
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW  torch.Size([8, 3, 352, 352])
        labels = sample_batched['labels'].to(device)  # BxHxW torch.Size([8, 1, 352, 352])

        ###############           train the encoder decoder Start     ################################

        concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
        concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)
        # Forward pass
        encoded, decoded = ecn_model(concatenated_smoothed_batch)

        # loss for encoder decoder and add to current loss
        reconstruction_loss = criterion_reconstruction(decoded, concatenated_smoothed_batch)
        ecn_optimiser.zero_grad()
        reconstruction_loss.backward()
        ecn_optimiser.step()
        reconstruction_loss_avg.append(reconstruction_loss.item())

        ###############        train the encoder decoder End         ################################



        #########################           train Unet Start     ################################

        #recompute improved encoder without gradients
        with torch.no_grad():
            encoded, _ = ecn_model(concatenated_smoothed_batch)

        preds_list = model(images, encoded.detach())

        unet_loss = sum([criterion(preds, labels,l_w) for preds, l_w in zip(preds_list,l_weight[:len(preds_list)])]) # bdcn_loss

        optimizer.zero_grad()
        unet_loss.backward()
        optimizer.step()
        unet_loss_avg.append(unet_loss.item())

        ##########################           train Unet End     ################################

        # if (batch_id == 100 and tb_writer is not None):
        #     tmp_unet_loss = np.array(unet_loss_avg).mean()
        #     tmp_reconstruction_loss = np.array(reconstruction_loss_avg).mean()
        #     tb_writer.add_scalar('Tmp Unet loss', tmp_unet_loss, epoch)
        #     tb_writer.add_scalar('Tmp Reconstruction loss', tmp_reconstruction_loss, epoch)

        
        if (batch_id % 100 == 0):
            tmp_unet_loss = np.array(unet_loss_avg).mean()
            tmp_reconstruction_loss = np.array(reconstruction_loss_avg).mean()
            wandb.log({"Tmp Unet loss": tmp_unet_loss, "Tmp Reconstruction loss": tmp_reconstruction_loss})

            
        if batch_id % 5 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Unet Loss: {3} ECN Loss : {4}'.format(epoch, batch_id, \
                            len(dataloader), unet_loss.item(), reconstruction_loss.item() ))

        if batch_id % log_interval_vis == 0:
            res_data = []
            img = images.cpu().numpy()
            res_data.append(img[2])
            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])
            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)
            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data
            vis_imgs = cv2.resize(vis_imgs, (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Unet Loss: {3} ECN Loss : {4}'.format(epoch, batch_id, \
                                        len(dataloader), unet_loss.item(), reconstruction_loss.item())
            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    reconstruction_loss_avg = np.array(reconstruction_loss_avg).mean()
    unet_loss_avg = np.array(unet_loss_avg).mean()
    return reconstruction_loss_avg , unet_loss_avg


def validate_one_epoch(epoch, dataloader, model, ecn_model, device, output_dir, arg = None):
    # XXX This is not really validation, but testing

    # Put model in eval mode
    model.eval()
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
            concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)
            # Forward pass
            encoded, _ = ecn_model(concatenated_smoothed_batch)

            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images, encoded)
            # print('pred shape', preds[0].shape)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,img_shape=image_shape,
                                     arg=arg)

def test(checkpoint_path, dataloader, model, ecn_model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Put model in evaluation mode
    model.eval()
    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
            concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)
            # Forward pass
            encoded, _ = ecn_model(concatenated_smoothed_batch)

            preds = model(images, encoded)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds, output_dir, file_names, image_shape, arg=args)
            torch.cuda.empty_cache()
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))

def testPich(checkpoint_path, dataloader, model, ecn_model, device, output_dir, args):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            # images2 = images[:, [1, 0, 2], :, :]  #GBR
            images2 = images[:, [2, 1, 0], :, :] # RGB

            concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
            concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)
            # Forward pass
            encoded, _ = ecn_model(concatenated_smoothed_batch)

            preds = model(images, encoded)

            concatenated_smoothed_batch2 = compute_Image_gradients(images2.cpu().numpy())
            concatenated_smoothed_batch2= torch.from_numpy(concatenated_smoothed_batch2).to(device)
            # Forward pass
            encoded2, _ = ecn_model(concatenated_smoothed_batch2)

            preds2 = model(images2, encoded2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg = args, 
                                     is_inchannel = True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UNet trainer.')
    parser.add_argument('--choose_test_data',
                        type = int,
                        default = -1,
                        help='Already set the dataset for testing choice: 0 - 8')
    parser.add_argument('--is_testing',
                        type = int,
                        default = 0,
                        help='Script in testing mode.')
    parser.add_argument('--resume',
                        type =  bool,
                        default = False,
                        help = 'use previous trained data')  # Just for test
    # ----------- test -------0--


    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] # max 8
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    # is_testing =0#  current test -352-SM-NewGT-2AugmenPublish

    # Training settings
    TRAIN_DATA = DATASET_NAMES[0] # BIPED=0, MDBD=6
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']


    # Data parameters
    parser.add_argument('--use_unet',
                        type=bool,
                        default=True,
                        help='True: use same 2 imgs changing channels')
    
    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input data.')
    
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    
    parser.add_argument('--output_dir',
                        type=str,
                        default='ecn_mycheckpoints',
                        help='the path to output the results.')
    
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    
    parser.add_argument('--test_data',
                        type = str,
                        choices = DATASET_NAMES,
                        default = TEST_DATA,
                        help = 'Name of the dataset.')
    
    parser.add_argument('--test_list',
                        type = str,
                        default = test_inf['test_list'],
                        help = 'Dataset sample indices list.')
    
    parser.add_argument('--train_list',
                        type = str,
                        default = train_inf['train_list'],
                        help = 'Dataset sample indices list.')
    
    parser.add_argument('--double_img',
                        type = bool,
                        default = False,
                        help = 'True: use same 2 imgs changing channels')  # Just for test
    
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='24/24_model.pth',# 4 6 7 9 14
                        help='Checkpoint path from which to restore model weights from.')
    
    parser.add_argument('--test_img_width',
                        type = int,
                        default = test_inf['img_width'],
                        help='Image width for testing.')
    
    parser.add_argument('--test_img_height',
                        type = int,
                        default = test_inf['img_height'],
                        help = 'Image height for testing.')
    
    parser.add_argument('--res_dir',
                        type = str,
                        default = 'result',
                        help = 'Result directory')
    
    parser.add_argument('--log_interval_vis',
                        type = int,
                        default = 50,
                        help='The number of batches to wait before printing test predictions.')

    parser.add_argument('--epochs',
                        type = int,
                        default = 25,
                        metavar = 'N',
                        help='Number of training epochs (default: 25).')
    
    parser.add_argument('--lr',
                        default = 1e-4,
                        type = float,
                        help='Initial learning rate.')
    
    parser.add_argument('--lr_ecn',
                        default = 1e-3,
                        type = float,
                        help='Initial learning rate.')
    
    parser.add_argument('--wd',
                        type=float,
                        default=1e-8,
                        metavar='WD',
                        help='weight decay (Good 1e-8) in TF1=0') # 1e-8 -> BIRND/MDBD, 0.0 -> BIPED
    
    parser.add_argument('--adjust_lr',
                        default=[10,15],
                        type=int,
                        help='Learning rate step size.') #[5,10]BIRND [10,15]BIPED/BRIND
    
    parser.add_argument('--batch_size',
                        type=int,
                        default = 8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    
    parser.add_argument('--workers',
                        default = 16 if IS_LINUX else 4,
                        # default = 4,
                        type = int,
                        help='The number of workers for the dataloaders.')
    
    parser.add_argument('--tensorboard',
                        type = bool,
                        default = False,
                        help='Use Tensorboard for logging.'),
    
    parser.add_argument('--img_width',
                        type = int,
                        default = 352,
                        help='Image width for training.') # BIPED 400 BSDS 352/320 MDBD 480
    
    parser.add_argument('--img_height',
                        type = int,
                        default = 352,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    
    parser.add_argument('--channel_swap',
                        default = [2, 1, 0],
                        type = int)
    
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default = [103.939, 116.779, 123.68, 137.86],
                        type = float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    #WANDB Setup to track hyper parameters
    wandb.init(
    # set the wandb project where this run will be logged
    project="Unet-Edge-Detection",
        # track hyperparameters and run metadata
        config={
        "unet_learning_rate": args.lr,
        "ecn_learning_rate" : args.lr_ecn,
        "architecture": "UNet + Encoder Decoder",
        "dataset": "MBIPED",
        "epochs": args.epochs,
        "batch_size train/val" : args.batch_size,
        "description" : "We seperate training the ECN from the Unet and prevent gradients from going back through both networks"
        }
    )


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
    if not args.resume:
        ini_epoch = 0
    # import pdb;pdb.set_trace()
    if args.tensorboard and not args.is_testing:
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir = training_dir)
        # saving Model training settings
        training_notes = ['DexiNed, Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR='+ str(args.adjust_lr) + ' Loss Function= BDCNloss2. '
                          +'Trained on> '+args.train_data+' Tested on> '
                          +args.test_data+' Batch size= '+str(args.batch_size)+' '+str(time.asctime())]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # Instantiate model and move it to the computing device
    if not args.use_unet:
        model = DexiNed().to(device) #* 35215245
    else:
        model = UNet(n_channels = 3, n_classes = 1, bilinear = True).to(device) #* 17262977

    ecn_model = EncoderDecoder(input_channels = 7).to(device)
    
    #Add number of parameters to wandb hyper parameters
    wandb.config.num_unet_parameters = count_parameters(model)
    wandb.config.num_ECN_parameters = count_parameters(ecn_model)


    if not args.is_testing:
        if args.resume:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print('Training restarted from> ',checkpoint_path)
        dataset_train = BipedDataset(args.input_dir,
                                     img_width = args.img_width,
                                     img_height = args.img_height,
                                     mean_bgr = args.mean_pixel_values[0:3] if len(args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode = 'train',
                                     arg = args)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size = args.batch_size,
                                      shuffle = True,
                                      num_workers = args.workers)

    print('args.input_val_dir: ', args.input_val_dir)
    # dataset_val = TestDataset(args.input_val_dir,
    #                           test_data = args.test_data,
    #                           img_width = args.test_img_width,
    #                           img_height = args.test_img_height,
    #                           mean_bgr = args.mean_pixel_values[0:3] if len(args.mean_pixel_values) == 4 else args.mean_pixel_values,
    #                           test_list = args.test_list, arg = args)


    dataset_val = TestDataset(args.input_val_dir,
                              test_data = args.test_data,
                              img_width = args.test_img_width,
                              img_height = args.test_img_height,
                              mean_bgr = args.mean_pixel_values[0:3] if len(args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list = args.test_list,
                              arg = args)
    dataloader_val = DataLoader(dataset_val,
                                batch_size = args.batch_size,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if args.is_testing:
        output_dir = os.path.join(args.res_dir, args.train_data + "2" + args.test_data)
        print(f"output_dir: {output_dir}")
        if args.double_img:
            # predict twice an image changing channels, then mix those results
            testPich(checkpoint_path, dataloader_val, model, device, output_dir, args)
        else:
            test(checkpoint_path, dataloader_val, model, device, output_dir, args)
        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('DexiNed # of Parameters:')
        print(num_param)
        print('-------------------------------------------------------')
        return
    
    criterion = bdcn_loss2 # hed_loss2 #bdcn_loss2
    criterion_reconstruction = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    ecn_optimiser = optim.Adam(ecn_model.parameters(), lr = args.lr_ecn, weight_decay = args.wd)

    # Main training loop
    seed = 1021
    adjust_lr = args.adjust_lr
    lr2 = args.lr
    # import pdb;pdb.set_trace()
    for epoch in range(ini_epoch, args.epochs):
        if epoch % 7 == 0:
            seed = seed + 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # Create output directories
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = lr2 * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
        output_dir_epoch = os.path.join(args.output_dir, args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch, exist_ok = True)
        os.makedirs(img_test_dir, exist_ok = True)
 
        val_avg_precision, val_avg_recall, val_ap,\
              val_reconstruction_loss_avg, val_unet_loss_avg = \
                validate_avg_precison(dataloader_val, 
                                        model,
                                        ecn_model, 
                                        criterion,
                                        criterion_reconstruction,
                                        device)
        
        avg_unet_loss, avg_ecn_loss = train_one_epoch(epoch,
                                        dataloader_train,
                                        model,
                                        ecn_model,
                                         criterion,
                                        criterion_reconstruction,
                                        optimizer,
                                        ecn_optimiser,
                                        device,
                                        args.log_interval_vis,
                                        tb_writer,
                                        args = args)
        # validate_one_epoch(epoch,
        #                    dataloader_val,
        #                    model,
        #                    device,
        #                    img_test_dir,
        #                    arg=args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        # if tb_writer is not None:
        #     tb_writer.add_scalar('loss', avg_loss, epoch+1)

        wandb.log({ "Unet loss": avg_unet_loss, 
                    "Ecn_loss": avg_ecn_loss,
                    "validation_avg_precision": val_avg_precision,
                    "validation_avg_recall" : val_avg_recall,
                    "avg_precision_formula" : val_ap, 
                    "validation_reconstruction_loss_avg" : val_reconstruction_loss_avg,
                    "unet_loss_avg" : val_unet_loss_avg,
                    "Current Learning rate Unet" : optimizer.param_groups[0]['lr']})
        
        print('Current learning rate> ', optimizer.param_groups[0]['lr'])
    
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('DexiNed, # of Parameters:')
    print(num_param)
    print('-------------------------------------------------------')

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
