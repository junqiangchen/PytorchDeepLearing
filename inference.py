import torch
import os
from model import *
from dataprocess.utils import file_name_path
import SimpleITK as sitk

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def inferencemutilunet3dtest():
    newSize = (112, 112, 128)
    Unet3d = MutilUNet3dModel(image_depth=128, image_height=112, image_width=112, image_channel=1, numclass=1,
                              batch_size=1, loss_name='MutilFocalLoss', inference=True,
                              model_path=r'log\MutilUNet3d\focalloss\BinaryVNet2dSegModel.pth')
    datapath = r"F:\MedicalData\(ok)2022KiPA\dataset\test\image"
    makspath = r"F:\MedicalData\(ok)2022KiPA\dataset\test\label"
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        sitk_image = sitk.ReadImage(imagepathname)
        sitk_mask = Unet3d.inference(sitk_image, newSize)
        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(sitk_mask, maskpathname)


if __name__ == '__main__':
    inferencemutilunet3dtest()
