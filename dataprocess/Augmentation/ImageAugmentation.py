from ..Augmentation.images_masks_3dtransform import ImageDataGenerator3D
import pandas as pd
import numpy as np

'''
Feature Standardization
standardize pixel values across the entire dataset
ZCA Whitening
A whitening transform of an image is a linear algebra operation that reduces the redundancy in the matrix of pixel images.
Less redundancy in the image is intended to better highlight the structures and features in the image to the learning algorithm.
Typically, image whitening is performed using the Principal Component Analysis (PCA) technique.
More recently, an alternative called ZCA (learn more in Appendix A of this tech report) shows better results and results in
transformed images that keeps all of the original dimensions and unlike PCA, resulting transformed images still look like their originals.
Random Rotations
sample data may have varying and different rotations in the scene.
Random Shifts
images may not be centered in the frame. They may be off-center in a variety of different ways.
RESCALE
对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0- 1之间，通常为1 / 255;
Random Flips
improve performance on large and complex problems is to create random flips of images in your training data.
fill_mode: 填充像素, 出现在旋转或平移之后．
'''


class DataAug3D(object):
    '''
        transform Image and Mask together
        '''

    def __init__(self, rotation=5, width_shift=0.01, height_shift=0.01, depth_shift=0.01, zoom_range=0.01,
                 rescale=1.1, horizontal_flip=True, vertical_flip=False, depth_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator3D(rotation_range=rotation, width_shift_range=width_shift,
                                              height_shift_range=height_shift, depth_shift_range=depth_shift,
                                              zoom_range=zoom_range,
                                              rescale=rescale, horizontal_flip=horizontal_flip,
                                              vertical_flip=vertical_flip, depth_flip=depth_flip,
                                              fill_mode='nearest')

    def __ImageMaskTranform(self, images_path, index, number, maxvalue=255):
        # reshape to be [samples][depth][width][height][channels]
        imagesample = np.load(images_path[0])
        srcimages = np.zeros((imagesample.shape[0], imagesample.shape[1], imagesample.shape[2]))
        srcimage = imagesample.reshape([1, srcimages.shape[0], srcimages.shape[1], srcimages.shape[2], 1])

        masksample = np.load(images_path[1])
        srcmasks = np.zeros((masksample.shape[0], masksample.shape[1], masksample.shape[2]))
        srcmask = masksample.reshape([1, srcmasks.shape[0], srcmasks.shape[1], srcmasks.shape[2], 1])

        i = 0
        for batchx, batchy in self.__datagen.flow(srcimage, srcmask):
            i += 1
            batch1 = batchx[0, :, :, :, :]
            batch2 = batchy[0, :, :, :, :]

            npy_path1 = self.aug_path + 'Image/' + str(index) + '_' + str(i) + ".npy"
            batch1 = batch1.reshape([srcimages.shape[0], srcimages.shape[1], srcimages.shape[2]])
            np.save(npy_path1, batch1)

            npy_path2 = self.aug_path + 'Mask/' + str(index) + '_' + str(i) + ".npy"
            batch2 = batch2.reshape([srcmasks.shape[0], srcmasks.shape[1], srcmasks.shape[2]])
            batch2 = batch2.astype('uint8')
            np.save(npy_path2, batch2)
            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, number=100, aug_path=None):
        csvXdata = pd.read_csv(filepathX)
        data = csvXdata.iloc[:, :].values
        self.aug_path = aug_path
        for index in range(data.shape[0]):
            # For images
            images_path = data[index]
            self.__ImageMaskTranform(images_path, index, number)
