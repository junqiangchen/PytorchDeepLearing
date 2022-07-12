import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


# define datasetModelClassifywithnpy class wiht npy
class datasetModelClassifywithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(1, 64, 128, 128)):
        super(datasetModelClassifywithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]
        image = np.reshape(image, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelClassifywithopencv class with npy
class datasetModelClassifywithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelClassifywithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]))
        # normalization image to 0-1
        image = image / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithnpy class wiht npy
class datasetModelSegwithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelSegwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]
        image = np.reshape(image, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithopencv class with npy
class datasetModelSegwithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelSegwithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]))
        # normalization image to 0-1
        image = image / 255.
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = cv2.imread(labelpath, 0)
        label = cv2.resize(label, (self.targetsize[1], self.targetsize[2]))
        label = label / 255.
        # transpose (H,W,C) order to (C,H,W) order
        label = np.reshape(label, (H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}
