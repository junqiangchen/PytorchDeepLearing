from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path

image_pre = ".nii.gz"
mask_pre = ".nii.gz"


def getImageSizeandSpacing(aorticvalve_path):
    """
    get image and spacing
    :return:
    """
    file_path_list = file_name_path(aorticvalve_path, False, True)
    size = []
    spacing = []
    for subsetindex in range(len(file_path_list)):
        if mask_pre in file_path_list[subsetindex]:
            mask_name = file_path_list[subsetindex]
            mask_gt_file = aorticvalve_path + "/" + mask_name
            src = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
            imageSize = src.GetSize()
            imageSpacing = src.GetSpacing()
            size.append(np.array(imageSize))
            spacing.append(np.array(imageSpacing))
            print("image size,image spacing:", (imageSize, imageSpacing))
    print("mean size,mean spacing:", (np.mean(np.array(size), axis=0), np.mean(np.array(spacing), axis=0)))


if __name__ == "__main__":
    aorticvalve_path = r"F:\MedicalData\(ok)2022KiPA\dataset\train\label"
    getImageSizeandSpacing(aorticvalve_path)
