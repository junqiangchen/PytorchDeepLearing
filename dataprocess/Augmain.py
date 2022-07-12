from dataprocess.Augmentation.ImageAugmentation import DataAug3D

if __name__ == '__main__':
    aug = DataAug3D(rotation=5, width_shift=0.01, height_shift=0.01, depth_shift=0, zoom_range=0,
                    vertical_flip=True, horizontal_flip=True)
    aug.DataAugmentation('data/traindata.csv', 15, aug_path='D:\challenge\data\KiPA2022\\trainstage\\augtrain/')
