import albumentations as A
import pandas as pd
import cv2


class Segmenation_Aug(object):
    def __init__(self):
        self.transform_compose = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                     A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                     A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                     ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),  # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),  # 随机明亮对比度
        ])

    def __ImageMaskTransform(self, imagedata, index, number):
        image_path = imagedata[0]
        mask_path = imagedata[1]
        src_image = cv2.imread(image_path)
        src_mask = cv2.imread(mask_path, 0)
        for i in range(number):
            transformed = self.transform_compose(image=src_image, mask=src_mask)
            image = transformed["image"]
            mask = transformed["mask"]
            cv2.imwrite(self.aug_path + '/Image/' + str(index) + '_' + str(i) + '.bmp', image)
            cv2.imwrite(self.aug_path + '/Mask/' + str(index) + '_' + str(i) + '.bmp', mask)

    def DataAugmentation(self, filepath, number=100, aug_path=None):
        csvXdata = pd.read_csv(filepath)
        data = csvXdata.iloc[:, :].values
        self.aug_path = aug_path
        for index in range(data.shape[0]):
            # For images
            imagedata = data[index]
            self.__ImageMaskTransform(imagedata, index, number)


if __name__ == '__main__':
    seg_aug = Segmenation_Aug()
    seg_aug.DataAugmentation('data/train.csv', 10,
                             aug_path=r'E:\challenge\data\iChallenge-PM2022\2023.7.31\all_data\aug_train')
