import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.dice_score import multiclass_dice_coeff, dice_coeff
from torchvision import transforms
from PIL import Image
import cv2

from model import *
from loss import Loss
from util import make_optimizer, calc_psnr, summary
import os
from util import vis_segments_AMZ
from util import vis_segments_CamVid

class Operator:
    def __init__(self, config, ckeck_point):
        self.config = config
        self.epochs = config.epochs
        self.uncertainty = config.uncertainty
        self.data_name = config.data_name
        self.test_interval = config.test_interval
        self.ckpt = ckeck_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)

        # set model, criterion, optimizer
        self.model = Model(config)
        self.n_classes = config.n_classes
        summary(self.model, config_file=self.ckpt.config_file)

        # set criterion, optimizer
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)

        # load ckpt, model, optimizer
        if self.ckpt.exp_load is not None or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)
        # save 

        self.save_dir = ckeck_point.save_dir

    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        train_batch_num = len(data_loader['train'])
        show_interval = train_batch_num // 5

        for epoch in range(last_epoch, self.epochs):
            for batch_idx, batch_data in enumerate(data_loader['train']):
                batch_input = batch_data['image']
                batch_label = batch_data['mask'] # //60 for amz big data

                # batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                
                # forward
                batch_results = self.model(batch_input)
                loss = self.criterion(batch_results, batch_label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % show_interval == 0:
                    print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:5f}'
                        .format(epoch, self.config.epochs,
                                batch_idx, train_batch_num,
                                loss.item()))

                # use tensorboard
                if self.tensorboard:     
                    batch_pred_labels = torch.softmax(batch_results['mean'], dim=1).argmax(dim=1)
                    if self.config.data_name == "AMZ":
                        batch_label_imgs = vis_segments_CamVid(batch_label.detach().long())
                        batch_pred_imgs = vis_segments_CamVid(batch_pred_labels.detach().long())
                    elif self.config.data_name == "CamVid":
                        batch_label_imgs = vis_segments_CamVid(batch_label.detach().long())
                        batch_pred_imgs = vis_segments_CamVid(batch_pred_labels.detach().long())
                    current_global_step = self.ckpt.step()
                    self.summary_writer.add_scalar('train/loss',
                                                   loss, current_global_step)
                    self.summary_writer.add_images("train/input_img",
                                                   batch_input,
                                                   current_global_step)

                    self.summary_writer.add_images("train/label_img",
                                                   batch_label_imgs,
                                                   current_global_step)
                    self.summary_writer.add_images("train/mean_img",
                                                   batch_pred_imgs,
                                                   current_global_step)

            # use tensorboard
            if self.tensorboard:
                print(self.optimizer.get_lr(), epoch)
                self.summary_writer.add_scalar('epoch_lr',
                                               self.optimizer.get_lr(), epoch)

            # test model & save model
            self.optimizer.schedule()
            self.save(self.ckpt, epoch)

            if epoch % self.test_interval == 0: # do test 
                self.test(data_loader)
                self.model.train()

        self.summary_writer.close()

    def test(self, data_loader):

        with torch.no_grad():
            self.model.eval()


            total_dice_score = 0
            dice_score_list = []
            test_batch_num = len(data_loader['test'])
            show_interval = test_batch_num // 5
            for batch_idx, batch_data in enumerate(data_loader['test']):

                batch_input = batch_data['image']
                batch_label = batch_data['mask'] # //60 for amz new data

                # batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                # forward
                batch_results = self.model(batch_input)
                mask_true = F.one_hot(batch_label, self.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(batch_input.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()

                current_dice =  multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                dice_score_list.append(current_dice)
                total_dice_score = sum(dice_score_list) / len(dice_score_list)


                if batch_idx % show_interval == 0:
                    print("Test iter: {:03d}/{:03d}, Total: {:5f}, Current: {:05f}".format(
                        batch_idx, test_batch_num,
                        total_dice_score, dice_score_list[batch_idx]))

                # use tensorboard
                if self.tensorboard:     
                    batch_pred_labels = torch.softmax(batch_results['mean'], dim=1).argmax(dim=1)
                    if self.config.data_name == "AMZ":
                        batch_label_imgs = vis_segments_CamVid(batch_label.detach().long())
                        batch_pred_imgs = vis_segments_CamVid(batch_pred_labels.detach().long())
                    elif self.config.data_name == "CamVid":
                        batch_label_imgs = vis_segments_CamVid(batch_label.detach().long())
                        batch_pred_imgs = vis_segments_CamVid(batch_pred_labels.detach().long())

                    self.summary_writer.add_scalar('test/dice_score',
                                                total_dice_score, self.ckpt.last_epoch)
                    self.summary_writer.add_images("test/input_img",
                                                batch_input, self.ckpt.last_epoch)

                    self.summary_writer.add_images("test/label_img",
                                                    batch_label_imgs,
                                                self.ckpt.last_epoch)

                    self.summary_writer.add_images("test/mean_img",
                                                    batch_pred_imgs,
                                                    self.ckpt.last_epoch)
                    if self.uncertainty == 'epistemic' or self.uncertainty == 'combined':
                        # do the colormap transform
                        _batch_size = (batch_label).size(dim=0)
                        # make a batch for tensorboard
                        e_var_tensor = torch.zeros_like(batch_label_imgs)
                        for index_i in range(_batch_size):
                            e_var = (batch_results['e_var'][index_i].float().cpu().numpy() * 255).astype('uint8')                         
                            heatmap_e_var = cv2.applyColorMap(e_var,cv2.COLORMAP_JET)
                            heatmap_e_var = heatmap_e_var[:,:,::-1] # bgr to rgb
                            e_var_tensor[index_i] = torch.from_numpy(heatmap_e_var.copy()).permute(2,0,1) # channel to first
                        self.summary_writer.add_images("test/epistemic_var",
                                e_var_tensor,
                                self.ckpt.last_epoch)

                    if self.uncertainty == 'aleatoric' or self.uncertainty == 'combined':
                        _batch_size = (batch_label).size(dim=0)
                        # make a batch for tensorboard
                        a_var_tensor = torch.zeros_like(batch_label_imgs)
                        for index_i in range(_batch_size):
                            a_var = (batch_results['a_var'][index_i].float().cpu().numpy() * 255).astype('uint8')      
                            heatmap_a_var = cv2.applyColorMap(a_var,cv2.COLORMAP_JET)  
                            heatmap_a_var = heatmap_a_var[:,:,::-1] # bgr to rgb
                            a_var_tensor[index_i] = torch.from_numpy(heatmap_a_var.copy()).permute(2,0,1) # channel to first
                        self.summary_writer.add_images("test/aleatoric_var",
                                a_var_tensor,
                                self.ckpt.last_epoch)

                    # if you want to save images instead of look into tensorboard you can use following
                    _batch_size = (batch_label).size(dim=0)
                    for index_i in range(_batch_size):
                        input_filename = os.path.join(self.save_dir,str(batch_idx).zfill(3) + '_batch_' + str(index_i)+'_epoch_'+str(self.ckpt.last_epoch) +'_input.png' )
                        mask_filename = os.path.join(self.save_dir,str(batch_idx).zfill(3) + '_batch_' + str(index_i)+'_epoch_'+str(self.ckpt.last_epoch) + '_label.png' )
                        pred_filename = os.path.join(self.save_dir,str(batch_idx).zfill(3) + '_batch_' + str(index_i)+'_epoch_'+str(self.ckpt.last_epoch) + '_pred.png' )
                        
                        
                        input_img_i = batch_input[index_i].permute((1,2,0)).float().cpu().numpy() * 255
                        mask_save_i = batch_label_imgs[index_i].permute((1,2,0)).float().cpu().numpy()
                        pred_save_i = batch_pred_imgs[index_i].permute((1,2,0)).float().cpu().numpy()
                        cv2.imwrite(input_filename,input_img_i)
                        cv2.imwrite(mask_filename,mask_save_i)
                        cv2.imwrite(pred_filename,pred_save_i)
                        if self.uncertainty == 'epistemic' or self.uncertainty == 'combined':
                            e_var_filename = os.path.join(self.save_dir,str(batch_idx).zfill(3) + '_batch_' + str(index_i)+'_epoch_'+str(self.ckpt.last_epoch) + 'e_var.png' )
                            heatmap_e_var = e_var_tensor[index_i].permute((1,2,0)).float().cpu().numpy().astype('uint8')
                            heatmap_e_var = heatmap_e_var[:,:,::-1] # rgb to bgr
                            cv2.imwrite(e_var_filename,heatmap_e_var)
                        if self.uncertainty == 'aleatoric' or self.uncertainty == 'combined':
                            a_var_filename = os.path.join(self.save_dir,str(batch_idx).zfill(3) + '_batch_' + str(index_i)+'_epoch_'+str(self.ckpt.last_epoch) + 'a_var.png' )
                            heatmap_a_var = a_var_tensor[index_i].permute((1,2,0)).float().cpu().numpy().astype('uint8')
                            heatmap_a_var = heatmap_a_var[:,:,::-1] # rgb to bgr
                            cv2.imwrite(a_var_filename,heatmap_a_var)
                        
            



    def mask_to_image(self,mask: np.ndarray):
        if mask.ndim == 2:
            return Image.fromarray((mask * 255).astype(np.uint8))
        elif mask.ndim == 3:
            return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model
        self.optimizer.load(ckpt) # load optimizer

    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt) # save optimizer:


