from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, define_Z, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
from utils import _gradient_penalty, tv_loss, extract_features, calc_Content_Loss, praf_metric, dice_metric, jaccard_metric
from metric import metrics

# Training settings
parser = argparse.ArgumentParser(description='WEGAN')
parser.add_argument('--dataset', type=str, default='MICROSCOPY', help='which dataset')
parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='location of datasets')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--noise_nc', type=int, default=128, help='input noise dimension')
parser.add_argument('--image_size', type=int, default=512, help='training image size')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output mask channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--loss', type=str, default='wgan-gp', help='lsgan or wgan-gp')
parser.add_argument('--dis', type=str, default='patch', help='use patch/pixel gan architecture')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=5, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=35, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--sparse_w', type=float, default=0.0006, help='weight of sparse penalty')
parser.add_argument('--content_w_all', type=float, default=800.0, help='weight of perceptual loss')
parser.add_argument('--BCE_w', type=float, default=3000.0, help='weight of BCE loss')
parser.add_argument('--content_layers', type=int, nargs='+', default=[15], help='layer indices to extract content features')
parser.add_argument('--vgg-flag', type=str, default='vgg16', help='VGG flag for calculating losses')
opt = parser.parse_args()

print(opt)
opt.cuda = True

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.dataset, opt.dataset_dir)
test_set = get_test_set(opt.dataset, opt.dataset_dir)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:3" if opt.cuda else "cpu")

print('===> Building models')
net_g1 = define_G(opt.noise_nc, opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_id=device)
net_g2 = define_G(opt.noise_nc, opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_id=device)
if opt.loss == 'wgan-gp':
    net_d1 = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
    net_d2 = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
elif opt.loss == 'lsgan':
    net_d1 = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
    net_d2 = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
else:
    print('wrong input')
    assert False

# Loss network
loss_network = torchvision.models.__dict__[opt.vgg_flag](pretrained=True).features.to(device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
criterionBCE = nn.BCELoss().to(device)

# setup optimizer
optimizer_g1 = optim.Adam(net_g1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d1 = optim.Adam(net_d1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

net_g1_scheduler = get_scheduler(optimizer_g1, opt)
net_d1_scheduler = get_scheduler(optimizer_d1, opt)

print('===> saving directory')
path_name = '{}_{}_{}'.format(opt.dataset, opt.dis, opt.loss)
step = 0
writer = SummaryWriter('runs/{}/'.format(path_name))
image_path = 'result/{}'.format(path_name)
if not os.path.isdir(image_path):
    os.mkdir(image_path)


# The first stage: update the mask and background networks simultaneously
for epoch in range(opt.epoch_count, opt.niter + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        noise_vec = torch.randn(1, opt.noise_nc).to(device)
        real = batch.to(device)
        mask, back = net_g1(noise_vec, real)

        sparse_loss = torch.norm(mask, p=1) # calculate the sparse loss on the generated mask

        batch_size = real.shape[0]
        back = back.repeat(batch_size, 1, 1, 1)
        fake = (mask * real) + ((1-mask) * back)
        fake = torch.clamp(fake, min=-1, max=1)

        real_ctt = real.repeat(1, 3, 1, 1)
        fake_ctt = fake.repeat(1, 3, 1, 1)
        real_content_features = extract_features(loss_network, real_ctt, opt.content_layers)
        fake_content_features = extract_features(loss_network, fake_ctt, opt.content_layers)

        content_loss_all = calc_Content_Loss(real_content_features, fake_content_features) # calculate the content loss between the input and output

        ######################
        # (1) Update D network
        ######################
        optimizer_d1.zero_grad()
        
        # train with fake
        pred_fake = net_d1.forward(fake.detach())
        
        # train with real
        pred_real = net_d1.forward(real)
        
        # Combined D loss
        if opt.loss == 'wgan-gp':
            gradient_penalty = _gradient_penalty(net_d1, real, fake, opt.cuda, device)
            loss_d = pred_fake.mean() - pred_real.mean() + gradient_penalty
        else:
            loss_d_fake = criterionGAN(pred_fake, False)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
    
        optimizer_d1.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g1.zero_grad()

        # G(A) should fake the discriminator
        pred_fake = net_d1.forward(fake)
        if opt.loss == 'wgan-gp':
            loss_g = - pred_fake.mean() + opt.sparse_w*sparse_loss + opt.content_w_all*content_loss_all
        else:
            loss_g = criterionGAN(pred_fake, True) + opt.sparse_w*sparse_loss + opt.content_w_all*content_loss_all
        
        loss_g.backward()

        optimizer_g1.step()

#        if iteration%5 == 0:
#            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_SP: {:.4f} Loss_C_all: {:.4f}".format(
#                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), sparse_loss.item(), content_loss_all.item()))
        
        info = {'loss_G': loss_g, 'loss_D': loss_d, 'loss_SP': sparse_loss, 'loss_C_all': content_loss_all}

        writer.add_scalars("losses", info, (step))
        step += 1

    
    # test
    if epoch%1 == 0:
        prediction = []
        truth = []
        for iteration, (image, label) in enumerate(testing_data_loader, 1):
            with torch.no_grad():
                noise_vec = torch.randn(1, opt.noise_nc).to(device)
                image = image.to(device)
                label = label.to(device)
                mask, back = net_g1(noise_vec, image)

                prediction.append(mask.round().float())
                truth.append(label.float())

                batch_size = image.shape[0]

                back = back.repeat(batch_size, 1, 1, 1)

                fake = (mask * image) + ((1-mask) * back)
                fake = torch.clamp(fake, min=-1, max=1)

                mask = mask * 2 - 1
                label = label * 2 -1

                combine = torch.cat((image, mask, label, back, fake), dim=3, out=None)
                combine = torch.squeeze(combine, 1)
                combine = combine.data.cpu().numpy()
                combine = (combine + 1) * 128
                combine = np.reshape(combine, (-1, opt.image_size*5))
                print("===> Testing:[{}]({}/{})".format(epoch, iteration, len(testing_data_loader)))
                if epoch%5 == 0:
                    cv2.imwrite('{}/{}_{}.jpg'.format(image_path, epoch, iteration), combine)

        prediction = torch.cat(prediction, dim=0)
        truth = torch.cat(truth, dim=0)
        prec, recall, acc, fscore = praf_metric(prediction, truth)
        dice = dice_metric(prediction, truth)
        jaccard = jaccard_metric(prediction, truth)
        print("===> Evaluation: Precision: {:.4f} Recall: {:.4f} Accuracy: {:.4f} F-score: {:.4f} Dice: {:.4f} Jaccard: {:.4f}".format(
                prec, recall, acc, fscore, dice, jaccard))

        truth_np = truth.detach().cpu().numpy().astype(np.float32)
        truth_np = np.squeeze(truth_np, axis = 1)
        prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
        prediction_np = np.squeeze(prediction_np, axis = 1)
        dice_orig, dice_post, dice, IOU, f1score, AUL = metrics(prediction_np, truth_np)
        print("===> NEW Evaluation: Dice_orig: {:.4f} Dice_post: {:.4f} Dice: {:.4f} IOU: {:.4f} F1score: {:.4f} AUL: {:.4f}".format(
                dice_orig, dice_post, dice, IOU, f1score, AUL))

    update_learning_rate(net_g1_scheduler, optimizer_g1)
    update_learning_rate(net_d1_scheduler, optimizer_d1)


    #checkpoint
    if epoch % 5 == 0:
        ckpt_path = os.path.join("checkpoint", path_name)
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        
        net_g1_model_out_path = "{}/netG1_model_epoch_{}.pth".format(ckpt_path, epoch)
        net_d1_model_out_path = "{}/netD1_model_epoch_{}.pth".format(ckpt_path, epoch)
        torch.save(net_g1, net_g1_model_out_path)
        torch.save(net_d1, net_d1_model_out_path)
        print("Checkpoint saved to {}".format(ckpt_path))

net_g1.eval()

# copy network states
net_g2.load_state_dict(net_g1.state_dict())
net_d2.load_state_dict(net_d1.state_dict())

# setup optimizer
optimizer_g2 = optim.Adam(net_g2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d2 = optim.Adam(net_d2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

net_g2_scheduler = get_scheduler(optimizer_g2, opt)
net_d2_scheduler = get_scheduler(optimizer_d2, opt)

# The second stage: fix the generator network, train the U-Net with the pseudo label generated by the generator network
for epoch in range(opt.niter + 1, opt.niter * 8 + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        noise_vec = torch.randn(1, opt.noise_nc).to(device)
        real = batch.to(device)
        mask_label, back = net_g1(noise_vec, real)
        mask, back = net_g2(noise_vec, real)
        mask = (mask + 1)/2

        # convert the generated mask to pseudo mask label
        mask_label = torch.clamp(mask_label, min=0, max=1)
        mask_label = mask_label*255
        mask_label = mask_label.detach().cpu().numpy().astype(np.uint8)
        for i in range(opt.batch_size):
            mask_i = mask_label[i, 0, :, :]
            ret2, mask_pseudo_label = cv2.threshold(mask_i, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_pseudo_label = mask_pseudo_label.astype(bool)
#            mask_pseudo_label = morphology.remove_small_objects(mask_pseudo_label, min_size=1000, connectivity=1)
            mask_pseudo_label = morphology.remove_small_holes(mask_pseudo_label, 32)
#            mask_pseudo_label = binary_fill_holes(mask_pseudo_label)
            mask_label[i, 0, :, :] = mask_pseudo_label
        mask_pseudo_label = torch.from_numpy(mask_label).float().to(device)

        # calculate the cross entropy loss
        loss_CE = criterionBCE(mask, mask_pseudo_label)

        sparse_loss = torch.norm(mask, p=1) # calculate the sparse loss on the generated mask

        batch_size = real.shape[0]
        back = back.repeat(batch_size, 1, 1, 1)
        fake = (mask * real) + ((1-mask) * back)
        fake = torch.clamp(fake, min=-1, max=1)

        real_ctt = real.repeat(1, 3, 1, 1)
        fake_ctt = fake.repeat(1, 3, 1, 1)
        real_content_features = extract_features(loss_network, real_ctt, opt.content_layers)
        fake_content_features = extract_features(loss_network, fake_ctt, opt.content_layers)

        content_loss_all = calc_Content_Loss(real_content_features, fake_content_features) # calculate the content loss between the input and output

        ######################
        # (1) Update D network
        ######################
        optimizer_d2.zero_grad()
        
        # train with fake
        pred_fake = net_d2.forward(fake.detach())
        
        # train with real
        pred_real = net_d2.forward(real)
        
        # Combined D loss
        if opt.loss == 'wgan-gp':
            gradient_penalty = _gradient_penalty(net_d2, real, fake, opt.cuda, device)
            loss_d = pred_fake.mean() - pred_real.mean() + gradient_penalty
        else:
            loss_d_fake = criterionGAN(pred_fake, False)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
    
        optimizer_d2.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g2.zero_grad()

        # G(A) should fake the discriminator
        pred_fake = net_d2.forward(fake)
        if opt.loss == 'wgan-gp':
            loss_g = - pred_fake.mean() + 6.0*opt.sparse_w*sparse_loss + 0.2*opt.content_w_all*content_loss_all + opt.BCE_w*loss_CE
        else:
            loss_g = criterionGAN(pred_fake, True) + 2.0*opt.sparse_w*sparse_loss + 0.2*opt.content_w_all*content_loss_all + opt.BCE_w*loss_CE
        
        loss_g.backward()

        optimizer_g2.step()

#        if iteration%5 == 0:
#            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_SP: {:.4f} Loss_C_all: {:.4f} Loss_BCE: {:.4f}".format(
#                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), sparse_loss.item(), content_loss_all.item(), loss_CE.item()))
        
        info = {'loss_G': loss_g, 'loss_D': loss_d, 'loss_SP': sparse_loss, 'loss_C_all': content_loss_all, 'loss_CE': loss_CE}
    
        writer.add_scalars("losses", info, (step))
        step += 1
    
    # test
    if epoch%1 == 0:
        prediction = []
        truth = []
        for iteration, (image, label) in enumerate(testing_data_loader, 1):
            with torch.no_grad():
                noise_vec = torch.randn(1, opt.noise_nc).to(device)
                image = image.to(device)
                label = label.to(device)
                mask, back = net_g2(noise_vec, image)
                mask = (mask + 1)/2

                prediction.append(mask.round().float())
                truth.append(label.float())

                batch_size = image.shape[0]

                back = back.repeat(batch_size, 1, 1, 1)

                fake = (mask * image) + ((1-mask) * back)
                fake = torch.clamp(fake, min=-1, max=1)

                mask = mask * 2 - 1
                label = label * 2 - 1

                combine = torch.cat((image, mask, label, back, fake), dim=3, out=None)
                combine = torch.squeeze(combine, 1)
                combine = combine.data.cpu().numpy()
                combine = (combine + 1) * 128
                combine = np.reshape(combine, (-1, opt.image_size*5))
                print("===> Testing:[{}]({}/{})".format(epoch, iteration, len(testing_data_loader)))

                if epoch%10 == 0:
                    cv2.imwrite('{}/{}_{}.jpg'.format(image_path, epoch, iteration), combine)

        prediction = torch.cat(prediction, dim=0)
        truth = torch.cat(truth, dim=0)
        prec, recall, acc, fscore = praf_metric(prediction, truth)
        dice = dice_metric(prediction, truth)
        jaccard = jaccard_metric(prediction, truth)
        print("===> Evaluation: Precision: {:.4f} Recall: {:.4f} Accuracy: {:.4f} F-score: {:.4f} Dice: {:.4f} Jaccard: {:.4f}".format(
                prec, recall, acc, fscore, dice, jaccard))

        truth_np = truth.detach().cpu().numpy().astype(np.float32)
        truth_np = np.squeeze(truth_np, axis = 1)
        prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
        prediction_np = np.squeeze(prediction_np, axis = 1)
        dice_orig, dice_post, dice, IOU, f1score, AUL = metrics(prediction_np, truth_np)
        print("===> NEW Evaluation: Dice_orig: {:.4f} Dice_post: {:.4f} Dice: {:.4f} IOU: {:.4f} F1score: {:.4f} AUL: {:.4f}".format(
                dice_orig, dice_post, dice, IOU, f1score, AUL))

    update_learning_rate(net_g2_scheduler, optimizer_g2)
    update_learning_rate(net_d2_scheduler, optimizer_d2)

    #checkpoint
    if epoch % 20 == 0:
        ckpt_path = os.path.join("checkpoint", path_name)
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        
        net_g2_model_out_path = "{}/netG2_model_epoch_{}.pth".format(ckpt_path, epoch)
        net_d2_model_out_path = "{}/netD2_model_epoch_{}.pth".format(ckpt_path, epoch)
        torch.save(net_g2, net_g2_model_out_path)
        torch.save(net_d2, net_d2_model_out_path)
        print("Checkpoint saved to {}".format(ckpt_path))
