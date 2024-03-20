from .dataset import datasetModelRegressionwithopencv
import torch.nn as nn
from .metric import calc_psnr, calc_ssim
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
import numpy as np
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
from networks import initialize_weights
import time
from tqdm import tqdm
from .visualization import plot_result, save_images2dregression
import cv2


class LUConv2d(nn.Module):
    def __init__(self, nchan, prob=0.5):
        super(LUConv2d, self).__init__()
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(nchan)
        self.drop = nn.Dropout2d(p=prob, inplace=True)

    def forward(self, x):
        out = self.relu1(self.drop(self.bn1(self.conv1(x))))
        return out


def _make_nConv2d(nchan, depth, prob=0.5):
    layers = []
    for _ in range(depth):
        layers.append(LUConv2d(nchan, prob=prob))
    return nn.Sequential(*layers)


class InputTransition2d(nn.Module):
    def __init__(self, inChans, outChans, prob=0.5):
        super(InputTransition2d, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inChans, outChans, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(outChans)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu1(self.drop(self.bn1(self.conv1(x))))
        # convert input to 16 channels
        x16 = self.relu1(self.drop(self.bn1(self.conv2(x))))
        # print("x16", x16.shape)
        # print("out:", out.shape)
        out = torch.add(out, x16)
        # assert 1>3
        return out


class DownTransition2d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.5):
        super(DownTransition2d, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm2d(outChans)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
        self.ops = _make_nConv2d(outChans, nConvs, prob)

    def forward(self, x):
        down = self.relu1(self.drop(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = torch.add(out, down)
        return out


class UpTransition2d(nn.Module):
    def __init__(self, inChans, outChans, nConvs, prob=0.5):
        super(UpTransition2d, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = nn.InstanceNorm2d(outChans)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=prob, inplace=True)
        self.ops = _make_nConv2d(outChans, nConvs, prob)
        self.conv = nn.Conv2d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx):
        out = self.relu(self.drop(self.bn(self.up_conv(x))))
        xcat = torch.cat((out, skipx), 1)
        xcat = self.relu(self.drop(self.bn(self.conv(xcat))))
        out = self.ops(xcat)
        # print(out.shape, xcat.shape)
        # assert 1>3
        out = torch.add(out, xcat)
        return out


class OutputTransition2d(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition2d, self).__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv = nn.Conv2d(inChans, outChans, kernel_size=1)
        self.relu = nn.Tanh()

    def forward(self, x):
        # print(x.shape) # 1, 16, 64, 128, 128
        # assert 1>3
        # convolve 16 down to 2 channels
        output = self.relu(self.conv(x))
        return output


class GeneratorUNet2d(nn.Module):
    """
    Vnet2d implement
    """

    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, image_channel, numclass, init_features=16):
        super(GeneratorUNet2d, self).__init__()
        self.image_channel = image_channel
        self.numclass = numclass
        self.features = init_features

        self.in_tr = InputTransition2d(self.image_channel, self.features, 0.5)

        self.down_tr32 = DownTransition2d(self.features, self.features * 2, 2, 0.5)
        self.down_tr64 = DownTransition2d(self.features * 2, self.features * 4, 3, 0.5)
        self.down_tr128 = DownTransition2d(self.features * 4, self.features * 8, 3, 0.5)
        self.down_tr256 = DownTransition2d(self.features * 8, self.features * 16, 3, 0.5)

        self.up_tr256 = UpTransition2d(self.features * 16, self.features * 8, 3, 0.5)
        self.up_tr128 = UpTransition2d(self.features * 8, self.features * 4, 3, 0.5)
        self.up_tr64 = UpTransition2d(self.features * 4, self.features * 2, 2, 0.5)
        self.up_tr32 = UpTransition2d(self.features * 2, self.features, 1, 0.5)

        self.out_tr = OutputTransition2d(self.features, self.numclass)

    def forward(self, x):
        # print("x.shape:", x.shape)
        out16 = self.in_tr(x)
        # print("out16.shape:", out16.shape) # 1, 16, 128, 128
        # assert 1>3
        out32 = self.down_tr32(out16)
        # print("out32.shape:", out32.shape) # 1, 32, 64, 64
        # assert 1>3
        out64 = self.down_tr64(out32)
        # print("out64.shape:", out64.shape) # 1, 64, 32, 32
        # assert 1>3
        out128 = self.down_tr128(out64)
        # print("out128.shape:", out128.shape) # 1, 128, 16, 16
        # assert 1>3
        out256 = self.down_tr256(out128)
        # print("out256.shape", out256.shape) # 1, 256, 8, 8
        # assert 1>3
        out = self.up_tr256(out256, out128)
        # print("out.shape:", out.shape)

        out = self.up_tr128(out, out64)
        # print("out:", out.shape)

        out = self.up_tr64(out, out32)
        # print("out:", out.shape)
        # assert 1>3
        out = self.up_tr32(out, out16)
        # print("last out:", out.shape)
        # assert 1>3
        out = self.out_tr(out)
        # print("out:", out.shape)
        return out


class Discriminator2d(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(Discriminator2d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = Discriminator2d._block(self.in_channels, self.features, name="enc1")
        self.encoder2 = Discriminator2d._block(self.features, self.features * 2, name="enc2", )
        self.encoder3 = Discriminator2d._block(self.features * 2, self.features * 4, name="enc3")
        self.encoder4 = Discriminator2d._block(self.features * 4, self.features * 8, name="enc4")
        self.encoder5 = Discriminator2d._block(self.features * 8, self.features * 16, name="enc5")
        self.bottleneck = Discriminator2d._block(self.features * 16, self.features * 32, name="bottleneck")

        self.avg = nn.Conv2d(self.features * 32, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.bottleneck(self.encoder5(self.encoder4(self.encoder3(self.encoder2(self.encoder1(x))))))
        x = self.avg(x)
        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=5,
                padding=2,
                stride=2,
                bias=False, ),),
            (name + "norm1", nn.InstanceNorm2d(features)),
            (name + "relu1", nn.LeakyReLU(0.2, inplace=True)),
        ]))


torch.autograd.set_detect_anomaly(True)


class Pixel2PixelGAN2dModel(object):
    """
    trick to train Pixel2PixelGAN:normalization input 0~1,use relu in gen and not use maxpooling,
    add graident penty,using leakrelu in dis
    train more dis than gen
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, inference=False,
                 model_path=None, num_cpu=4, use_cuda=True):

        self.batch_size = batch_size
        self.accuracyname = ['PSNR', 'SSIM']
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass
        self.num_cpu = num_cpu
        self.alpha = 100

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.generator = GeneratorUNet2d(self.image_channel, self.image_channel)
        self.discriminator = Discriminator2d(self.image_channel * 2, self.numclass)

        # self.generator = Generator(self.image_channel, self.image_channel)
        # self.discriminator = Discriminator()

        self.generator.to(device=self.device)
        self.discriminator.to(device=self.device)
        self._loss_function()

        self.optimizer_D = None
        self.optimizer_G = None
        self.showpixelvalue = None
        self.model_dir = None

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelRegressionwithopencv(images, labels,
                                                   targetsize=(self.image_channel, self.image_height,
                                                               self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _lr_decay_step(self, current_epoch):
        self.g_lr_decay.step(current_epoch)
        self.d_lr_decay.step(current_epoch)

    def _loss_function(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recon = torch.nn.L1Loss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname[0] == 'PSNR':
            psnr = calc_psnr(input, target)
        if accuracyname[1] == 'SSIM':
            ssim = calc_ssim(input, target)
        return psnr, ssim

    def _train_loop(self, train_loader, totalTrainDLoss, totalTrainGLoss, totalTrainAccu, totalTrainssim, trainshow, e):
        self.clear_GPU_cache()
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,D,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,D,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            ones = torch.ones_like(self.discriminator(y, x))
            zeros = torch.zeros_like(self.discriminator(y, x))
            #########################################################################################################
            #                                                     Generator                                         #
            #########################################################################################################
            fake_imgs = self.generator(x)
            fake_validity = self.discriminator(fake_imgs, x)
            gan_loss = self.criterion_GAN(fake_validity, ones)
            recon_loss = self.criterion_recon(fake_imgs, y)
            g_loss = gan_loss + self.alpha * recon_loss
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()
            #########################################################################################################
            #                                                     Discriminator                                     #
            #########################################################################################################
            fake_imgs = self.generator(x)
            real_validity = self.discriminator(y, x)
            fake_validity = self.discriminator(fake_imgs, x)
            loss_real = self.criterion_GAN(real_validity, ones)
            loss_fake = self.criterion_GAN(fake_validity, zeros)
            d_loss = (loss_real + loss_fake) / 2.0
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            psnr, ssim = self._accuracy_function(self.accuracyname, fake_imgs, y)
            if trainshow:
                # save_images
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images2dregression(x[0], fake_imgs[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainDLoss.append(d_loss.cpu().detach().numpy())
            totalTrainGLoss.append(g_loss.cpu().detach().numpy())
            totalTrainAccu.append(psnr.cpu().detach().numpy())
            totalTrainssim.append(ssim)

    def _validation_loop(self, val_loader, totalValidationDLoss, totalValidationGLoss, totalValiadtionAccu,
                         totalValiadtionssim, trainshow, e):
        self.clear_GPU_cache()
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                ones = torch.ones_like(self.discriminator(y, x))
                zeros = torch.zeros_like(self.discriminator(y, x))
                # make the predictions and calculate the validation loss
                fake_imgs = self.generator(x)
                # Real images
                real_validity = self.discriminator(y, x)
                # Fake images
                fake_validity = self.discriminator(fake_imgs, x)
                gan_loss = self.criterion_GAN(fake_validity, ones)
                recon_loss = self.criterion_recon(fake_imgs, y)
                g_loss = gan_loss + self.alpha * recon_loss
                loss_real = self.criterion_GAN(real_validity, ones)
                loss_fake = self.criterion_GAN(fake_validity, zeros)
                d_loss = (loss_real + loss_fake) / 2.0
                psnr, ssim = self._accuracy_function(self.accuracyname, fake_imgs, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images2dregression(x[0], fake_imgs[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationDLoss.append(d_loss.cpu().detach().numpy())
                totalValidationGLoss.append(g_loss.cpu().detach().numpy())
                totalValiadtionAccu.append(psnr.cpu().detach().numpy())
                totalValiadtionssim.append(ssim)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=0.0002):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        print(self.generator)
        print(self.discriminator)
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        # 1、initialize loss function and optimizer
        self.generator.apply(initialize_weights)
        self.discriminator.apply(initialize_weights)
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        # Set learning-rate decay
        self.g_lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=100, gamma=0.1)
        self.d_lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=100, gamma=0.1)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, batch_size=self.batch_size, shuffle=True)
        val_loader = self._dataloder(validationimage, validationmask, batch_size=self.batch_size, shuffle=True)
        # 3、initialize a dictionary to store training history
        H = {"train_Dloss": [], "train_Gloss": [], "train_accuracy": [], "train_ssim": [],
             "valdation_Dloss": [], "valdation_Gloss": [], "valdation_accuracy": [], "valdation_ssim": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        best_e = 0
        best_validation_d = 10000000000
        best_validation_g = 10000000000
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.generator.train()
            self.discriminator.train()
            # 4.2、initialize the total training and validation loss
            totalTrainDLoss = []
            totalTrainGLoss = []
            totalTrainAccu = []
            totalTrainssim = []
            totalValidationDLoss = []
            totalValidationGLoss = []
            totalValiadtionAccu = []
            totalValiadtionssim = []
            # 4.3、loop over the training set
            trainshow = True
            self._train_loop(train_loader, totalTrainDLoss, totalTrainGLoss, totalTrainAccu, totalTrainssim, trainshow,
                             e)
            self._lr_decay_step(e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.generator.eval()
            self.discriminator.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationDLoss, totalValidationGLoss, totalValiadtionAccu,
                                  totalValiadtionssim, trainshow, e)
            # 4.5、calculate the average training and validation loss
            avgTrainDLoss = np.mean(np.stack(totalTrainDLoss))
            avgValidationDLoss = np.mean(np.stack(totalValidationDLoss))
            avgTrainGLoss = np.mean(np.stack(totalTrainGLoss))
            avgValidationGLoss = np.mean(np.stack(totalValidationGLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            avgTrainssim = np.mean(np.stack(totalTrainssim))
            avgValidationssim = np.mean(np.stack(totalValiadtionssim))
            # 4.6、update our training history
            H["train_Dloss"].append(avgTrainDLoss)
            H["valdation_Dloss"].append(avgValidationDLoss)
            H["train_Gloss"].append(avgTrainGLoss)
            H["valdation_Gloss"].append(avgValidationGLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            H["train_ssim"].append(avgTrainssim)
            H["valdation_ssim"].append(avgValidationssim)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("train_Dloss: {:.5f},train_Gloss: {:.5f}，train_accuracy: {:.5f}，train_ssim: {:.5f}".format(
                avgTrainDLoss, avgTrainGLoss, avgTrainAccu, avgTrainssim))
            print(
                "valdation_Dloss: {:.5f}, valdation_Gloss: {:.5f}，validation accu: {:.5f},validation valdation_ssim: "
                "{:.5f}".format(avgValidationDLoss, avgValidationGLoss, avgValidationAccu, avgValidationssim))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/DLoss', avgTrainDLoss, e + 1)
            writer.add_scalar('Train/GLoss', avgTrainGLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Train/ssim', avgTrainssim, e + 1)
            writer.add_scalar('Valid/Dloss', avgValidationDLoss, e + 1)
            writer.add_scalar('Valid/Gloss', avgValidationGLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.add_scalar('Valid/ssim', avgValidationssim, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if best_validation_d > avgValidationDLoss:
                best_validation_d = avgValidationDLoss
                discriminator_PATH = os.path.join(model_dir, "discriminator_best.pth")
                torch.save(self.discriminator.state_dict(), discriminator_PATH)
                best_e = e
            if best_validation_g > avgValidationGLoss:
                best_validation_g = avgValidationGLoss
                generator_PATH = os.path.join(model_dir, "generator_best.pth")
                torch.save(self.generator.state_dict(), generator_PATH)
                best_e = e
            generator_PATH = os.path.join(model_dir, "generator.pth")
            discriminator_PATH = os.path.join(model_dir, "discriminator.pth")
            torch.save(self.generator.state_dict(), generator_PATH)
            torch.save(self.discriminator.state_dict(), discriminator_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
            # 4.10、early stopping
            if abs(best_e - e) > epochs // 3:
                break
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(self.model_dir, H["train_Dloss"], H["train_Gloss"], "train_Dloss", "train_Gloss", "trainloss")
        plot_result(self.model_dir, H["valdation_Dloss"], H["valdation_Gloss"], "valdation_Dloss", "valdation_Gloss",
                    "validationloss")
        plot_result(self.model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy",
                    "valdation_accuracy",
                    "accuracy")
        plot_result(self.model_dir, H["train_ssim"], H["valdation_ssim"], "train_ssim", "valdation_ssim", "ssim")
        self.clear_GPU_cache()

    def predict(self, full_img):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.generator.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.generator(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
            return full_mask_np

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = imageresize / 255.
        out_mask = self.predict(imageresize)
        out_mask = out_mask * 255.
        out_mask = np.clip(out_mask, 0, 255).astype('uint8')
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
