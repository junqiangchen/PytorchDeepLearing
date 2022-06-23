import pandas as pd
from sklearn.model_selection import train_test_split
from dataprocess.utils import file_name_path
import torch
import os
from model import *
import cv2
import SimpleITK as sitk

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
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss')
    vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryVNet3d/CED',
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
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/BinaryUNet3d/CED',
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
                              batch_size=1, loss_name='MutilCrossEntropyLoss')
    vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilVNet3d/CE',
                        epochs=100, showwind=[8, 10])


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
                              batch_size=1, loss_name='MutilCrossEntropyLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilUNet3d/CE',
                        epochs=100, showwind=[8, 10])


def inferencebinaryvnet2d():
    data_dir = 'dataprocess/data/testseg.csv'
    csv_data = pd.read_csv(data_dir)
    valimages = csv_data.iloc[:, 0].values
    vallabels = csv_data.iloc[:, 1].values

    vnet2d = BinaryVNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=1, batch_size=8,
                               loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log/BinaryVNet2d/dice\BinaryVNet2dModel.pth')
    outpath = r"D:\cjq\data\GlandCeildata\test\pd2"
    for index in range(len(valimages)):
        image = cv2.imread(valimages[index], 0)
        mask = vnet2d.inference(image)
        cv2.imwrite(outpath + "/" + str(index) + ".png", mask)


def inferencemutilvnet2d():
    data_dir = 'dataprocess/data/testseg.csv'
    csv_data = pd.read_csv(data_dir)
    valimages = csv_data.iloc[:, 0].values
    vallabels = csv_data.iloc[:, 1].values

    vnet2d = MutilVNet2dModel(image_height=512, image_width=512, image_channel=1, numclass=2, batch_size=8,
                              loss_name='MutilDiceLoss', inference=True,
                              model_path=r'log/MutilVNet2d/dice\MutilVNet2d.pth')
    outpath = r"D:\cjq\data\GlandCeildata\test\pd2"
    for index in range(len(valimages)):
        image = cv2.imread(valimages[index], 0)
        mask = vnet2d.inference(image)
        cv2.imwrite(outpath + "/" + str(index) + ".png", mask)


def inferencebinaryvnet3d():
    data_dir = r'D:\cjq\data\Amos2022\ROIprocess\validation\Image'

    vnet3d = BinaryVNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\BinaryVNet3d\dice\BinaryVNet3d.pth')
    outpath = r"D:\cjq\data\Amos2022\ROIprocess\validation\Maskpd"
    image_files = file_name_path(data_dir, False, True)
    for index in range(len(image_files)):
        image_path = data_dir + '/' + image_files[index]
        sitkimage = sitk.ReadImage(image_path, sitk.sitkInt16)
        sitkmask = vnet3d.inference(sitkimage, newSize=(176, 112, 80))
        output_path = outpath + '/' + image_files[index]
        sitk.WriteImage(sitkmask, output_path)


def inferencemutilvnet3d():
    data_dir = r'D:\cjq\data\Amos2022\ROIprocess\validation\Image'

    vnet3d = MutilVNet3dModel(image_depth=80, image_height=112, image_width=176, image_channel=1, numclass=16,
                              batch_size=1, loss_name='MutilFocalLoss', inference=True,
                              model_path=r'log\MutilVNet3d\dice\MutilVNet3d.pth')
    outpath = r"D:\cjq\data\Amos2022\ROIprocess\validation\Maskpd"
    image_files = file_name_path(data_dir, False, True)
    for index in range(len(image_files)):
        image_path = data_dir + '/' + image_files[index]
        sitkimage = sitk.ReadImage(image_path, sitk.sitkInt16)
        sitkmask = vnet3d.inference(sitkimage, newSize=(176, 112, 80))
        output_path = outpath + '/' + image_files[index]
        sitk.WriteImage(sitkmask, output_path)


def trainmutilResNet2d():
    data_dir = 'dataprocess/data/mnisttrain.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 1].values
    trainlabels = csv_data.iloc[:, 0].values
    data_dir2 = 'dataprocess/data/mnistvalidation.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 1].values
    vallabels = csv_data2.iloc[:, 0].values
    resnet2d = MutilResNet2dModel(image_height=64, image_width=64, image_channel=1, numclass=10,
                                  batch_size=128, loss_name='MutilCrossEntropyLoss')
    resnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/MutilResNet2d/CE', epochs=50,
                          lr=0.001)


if __name__ == '__main__':
    # trainbinaryvnet2d()
    # trainbinaryunet2d()

    # trainmutilvnet2d()
    # trainmutilunet2d()

    # trainbinaryvnet3d()
    # trainbinaryunet3d()

    trainmutilvnet3d()
    trainmutilunet3d()

    # inferencebinaryvnet2d()
    # inferencemutilvnet2d()
    # inferencebinaryvnet3d()
    # inferencemutilvnet3d()

    # trainmutilResNet2d()
