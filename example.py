import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from model import *
import cv2

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainbinaryvnet2d():
    data_dir = 'dataprocess/data/trainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/testseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet2d = BinaryVNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=1, batch_size=8,
                               loss_name='BinaryFocalLoss')
    vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryVNet2d/focal', epochs=50)


def trainbinaryunet2d():
    data_dir = 'dataprocess/data/trainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/testseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet2d = BinaryUNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=1, batch_size=8,
                               loss_name='BinaryFocalLoss')
    unet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryUNet2d/focal', epochs=50)


def trainmutilvnet2d():
    data_dir = 'dataprocess/data/trainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/testseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet2d = MutilVNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=2, batch_size=8,
                              loss_name='MutilDiceLoss')
    vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilVNet2d/dice', epochs=50)


def trainmutilunet2d():
    data_dir = 'dataprocess/data/trainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/testseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet2d = MutilUNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=2, batch_size=8,
                              loss_name='MutilDiceLoss')
    unet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilUNet2d/dice', epochs=50)


def trainbinaryvnet3d():
    data_dir = 'dataprocess/data/amostrainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/amosvalidationseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet3d = BinaryVNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss')
    vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryVNet3d/dice',
                        epochs=50, showwind=[8, 10])


def trainbinaryunet3d():
    data_dir = 'dataprocess/data/amostrainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/amosvalidationseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = BinaryUNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryUNet3d/dice',
                        epochs=50, showwind=[8, 10])


def trainmutilvnet3d():
    data_dir = 'dataprocess/data/amostrainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/amosvalidationseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet3d = MutilVNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=16,
                              batch_size=1, loss_name='MutilFocalLoss')
    vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilVNet3d/focal',
                        epochs=50, showwind=[8, 10])


def trainmutilunet3d():
    data_dir = 'dataprocess/data/amostrainseg.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    data_dir2 = 'dataprocess/data/amosvalidationseg.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = MutilUNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=16,
                              batch_size=1, loss_name='MutilFocalLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilUNet3d/focal',
                        epochs=50, showwind=[8, 10])


def trainmutilResNet2d():
    data_dir = 'dataprocess/data/trainlabels.csv'
    csv_data = pd.read_csv(data_dir)
    images = csv_data.iloc[:, 0].values
    labels = csv_data.iloc[:, 1].values
    trainimages, valimages, trainlabels, vallabels = train_test_split(images, labels, test_size=0.2)
    unet3d = MutilResNet2dModel(image_height=256, image_width=256, image_channel=1, numclass=120,
                                batch_size=32, loss_name='MutilCrossEntropyLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilResNet2d/CE', epochs=50)


def inferencebinaryvnet2d():
    data_dir = 'dataprocess/data/testseg.csv'
    csv_data = pd.read_csv(data_dir)
    valimages = csv_data.iloc[:, 0].values
    vallabels = csv_data.iloc[:, 1].values

    vnet2d = BinaryVNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=1, batch_size=8,
                               loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log/BinaryVNet2d/BCED\BinaryVNet2dSegModel.pth')
    outpath = r"D:\cjq\data\GlandCeildata\test\pd2"
    for index in range(len(valimages)):
        image = cv2.imread(valimages[index], 0)
        mask = vnet2d.inference(image)
        cv2.imwrite(outpath + "/" + str(index) + ".png", mask)


if __name__ == '__main__':
    # trainbinaryvnet2d()
    # trainbinaryunet2d()

    # trainmutilvnet2d()
    # trainmutilunet2d()

    trainbinaryvnet3d()
    trainbinaryunet3d()

    # trainmutilvnet3d()
    # trainmutilunet3d()

    # inferencebinaryvnet2dseg()
