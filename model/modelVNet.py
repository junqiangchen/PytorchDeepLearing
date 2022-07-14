import torch
import torch.nn.functional as F
from networks.VNet2d import VNet2d
from networks.VNet3d import VNet3d
from .dataset import datasetModelSegwithopencv, datasetModelSegwithnpy
from torch.utils.data import DataLoader
from .losses import BinaryDiceLoss, BinaryFocalLoss, BinaryCrossEntropyLoss, BinaryCrossEntropyDiceLoss, \
    MutilDiceLoss, MutilFocalLoss, MutilCrossEntropyLoss
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .metric import dice_coeff, iou_coeff, multiclass_dice_coeff, multiclass_iou_coeff
from .visualization import plot_result, save_images2d, save_images3d
from pathlib import Path
import time
import os
import cv2
from dataprocess.utils import resize_image_itkwithsize, ConvertitkTrunctedValue
import SimpleITK as sitk
import multiprocessing
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class BinaryVNet2dModel(object):
    """
    Vnet2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='BinaryDiceLoss',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname is 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryVNet2dModel.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # save_images
                savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images2d(pred[0], y[0], savepath, pixelvalue=showpixelvalue)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    y[y != 0] = 1
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    # save_images
                    savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images2d(pred[0], y[0], savepath, pixelvalue=showpixelvalue)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            #lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height))
        imageresize = imageresize / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, image.shape)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilVNet2dModel(object):
    """
    Vnet2d with mutil class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MutilFocalLoss',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname is 'MutilDiceLoss':
            self.alpha[0] = 0.1
            return MutilDiceLoss(self.alpha)
        if lossname is 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilVNet2d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images2d(torch.argmax(pred[0], 0), torch.argmax(y[0], 0), savepath, pixelvalue=showpixelvalue)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    # save_images
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images2d(torch.argmax(pred[0], 0), torch.argmax(y[0], 0), savepath, pixelvalue=showpixelvalue)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
            # best_model_params = self.model.state_dict()
            # serialize best model to disk
            torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height))
        imageresize = imageresize / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, image.shape)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class BinaryVNet3dModel(object):
    """
    Vnet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryDiceLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname is 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryVNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # save_images
                savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images3d(pred[0], y[0], showwind, savepath, pixelvalue=showpixelvalue)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    y[y != 0] = 1
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    # save_images
                    savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images3d(pred[0], y[0], showwind, savepath, pixelvalue=showpixelvalue)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            #lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 100, -100, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilVNet3dModel(object):
    """
    VNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MutilFocalLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        self.gamma = 3

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname is 'MutilDiceLoss':
            self.alpha[0] = 0.1
            return MutilDiceLoss(alpha=self.alpha)
        if lossname is 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilVNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images3d(torch.argmax(pred[0], 0), torch.argmax(y[0], 0), showwind, savepath,
                              pixelvalue=showpixelvalue)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss)
                totalTrainAccu.append(accu)
            # 4.4、switch off autograd and loop over the validation set
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    # save_images
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images3d(torch.argmax(pred[0], 0), torch.argmax(y[0], 0), showwind, savepath,
                                  pixelvalue=showpixelvalue)
                    totalValidationLoss.append(loss)
                    totalValiadtionAccu.append(accu)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
            avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
            avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
            avgValidationAccu = torch.mean(torch.stack(totalValiadtionAccu))
            #lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
            H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
            H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 255
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 100, -100, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
