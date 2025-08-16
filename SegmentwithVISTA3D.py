import monai.transforms
import numpy as np
import torch
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.inferers import SlidingWindowInfererAdapt
from monai.networks.nets.vista3d import vista3d132
import SimpleITK as sitk
import time


class tumorVista3ddwithPoint_Inference():
    def __init__(self, point_inferer=True, ROI_SIZE=[128, 128, 128]):
        self.point_inferer = point_inferer  # use point based inferen
        self.ROI_SIZE = ROI_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # load model
        checkpoint_path = r"weights/vista3D_model.pth"
        self.model = vista3d132(in_channels=1)
        pretrained_ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(pretrained_ckpt, strict=True)

    def convert_clicks(self, alldata):
        # indexes = list(alldata.keys())
        # data = [alldata[i] for i in indexes]
        data = alldata
        B = len(data)  # Number of objects
        indexes = np.arange(1, B + 1).tolist()
        # Determine the maximum number of points across all objects
        max_N = max(len(obj["fg"]) + len(obj["bg"]) for obj in data)

        # Initialize padded arrays
        point_coords = np.zeros((B, max_N, 3), dtype=int)
        point_labels = np.full((B, max_N), -1, dtype=int)

        for i, obj in enumerate(data):
            points = []
            labels = []

            # Add foreground points
            for fg_point in obj["fg"]:
                points.append(fg_point)
                labels.append(1)

            # Add background points
            for bg_point in obj["bg"]:
                points.append(bg_point)
                labels.append(0)

            # Fill in the arrays
            point_coords[i, : len(points)] = points
            point_labels[i, : len(labels)] = labels

        return point_coords, point_labels, indexes

    def vista3d_infer(self, img_array, clicks):
        device = self.device
        point_coords, point_labels, indexes = self.convert_clicks(clicks)
        # slidingwindow
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            if not self.point_inferer:
                self.model.NINF_VALUE = 0  # set to 0 in case sliding window is used.
                # directly using slidingwindow inferer is not optimal.
                val_outputs = (
                        SlidingWindowInfererAdapt(
                            roi_size=self.ROI_SIZE,
                            sw_batch_size=1,
                            with_coord=True,
                            padding_mode="replicate",
                        )(
                            inputs=img_array.to(device),
                            transpose=True,
                            network=self.model.to(device),
                            point_coords=torch.from_numpy(point_coords).to(device),
                            point_labels=torch.from_numpy(point_labels).to(device),
                        )[0] > 0)
                final_outputs = torch.zeros_like(val_outputs[0], dtype=torch.float32)
                for i, v in enumerate(val_outputs):
                    final_outputs += indexes[i] * v
            else:
                # point based
                final_outputs = torch.zeros_like(img_array[0, 0], dtype=torch.float32)
                for i, v in enumerate(indexes):
                    val_outputs = (
                            point_based_window_inferer(
                                inputs=img_array.to(device),
                                roi_size=self.ROI_SIZE,
                                transpose=True,
                                with_coord=True,
                                predictor=self.model.to(device),
                                mode="gaussian",
                                sw_device=device,
                                device=device,
                                center_only=True,  # only crop the center
                                point_coords=torch.from_numpy(point_coords[[i]]).to(device),
                                point_labels=torch.from_numpy(point_labels[[i]]).to(device),
                            )[0] > 0)
                    final_outputs[val_outputs[0]] = v
                final_outputs = torch.nan_to_num(final_outputs)
            # save data
            seg_array = final_outputs.to(torch.float32).data.cpu().numpy()
            return seg_array

    def network_prediction(self, inputfilepath, unique_labs_list=None, lower_bound=-100, upper_bound=200):
        """
        :param inputfilepath: image path
        :param unique_labs_list: [[x1,y1,z1,x2,y2,z2,label],[x1,y1,z1,x2,y2,z2,label],[x1,y1,z1,x2,y2,z2,label]]
        :return:
        """
        if not (inputfilepath.endswith('.nii') or inputfilepath.endswith('.nii.gz') or inputfilepath.endswith('.mha')):
            print("文件格式不支持，仅支持 .nii, .nii.gz 和 .mha 格式")
            return False, None
        try:
            # preprocess
            nii_image = sitk.ReadImage(inputfilepath)
            img_array = sitk.GetArrayFromImage(nii_image)
            lower_bound, upper_bound = float(lower_bound), float(upper_bound)
            img_array = np.clip(img_array, lower_bound, upper_bound)
            img_array = torch.from_numpy(img_array)
            img_array = img_array.unsqueeze(0)
            img_array = monai.transforms.ScaleIntensityRangePercentiles(lower=1, upper=99,
                                                                        b_min=0, b_max=1, clip=True)(img_array)
            img_array = img_array.unsqueeze(0)  # add channel dim
            # preprocess clicks points
            clicks_dict = {}
            clicks_bg_list = []
            for box in unique_labs_list:
                x1, y1, z1, x2, y2, z2, label = box
                points = [[z1, y1, x1], [z2, y2, x2]]
                if label > 0:
                    if label not in clicks_dict:
                        clicks_dict[label] = []
                    clicks_dict[label].extend(points)
                else:
                    clicks_bg_list.extend(points)
            final_mask_array = np.zeros_like(sitk.GetArrayFromImage(nii_image))
            for label, fg_points in clicks_dict.items():
                clicks = [{"fg": fg_points, "bg": clicks_bg_list}]
                print(clicks)
                array_mask = self.vista3d_infer(img_array, clicks)
                final_mask_array[array_mask != 0] = label
            sitk_mask = sitk.GetImageFromArray(final_mask_array.astype('uint8'))
            sitk_mask.CopyInformation(nii_image)
            return True, sitk_mask
        except Exception as e:
            print(f"出现异常：{e}", inputfilepath)
            return False, None


def box_point_test_demo():
    input_image_path = r"C:\liver_image.nii.gz"
    output_mask_path = "liver_tumor_VISTA3d_point.nii.gz"
    box_list_2d_liver_tumor = [142, 344, 231, 143, 335, 229, 4]  # x1,y1,z1,x2,y2,z2,label
    box_list_3d_list = [box_list_2d_liver_tumor]
    start = time.time()
    vista3d_infer = tumorVista3ddwithPoint_Inference()
    _, sitk_mask = vista3d_infer.network_prediction(input_image_path, box_list_3d_list)
    end = time.time()
    print(end - start)
    sitk.WriteImage(sitk_mask, output_mask_path)


if __name__ == '__main__':
    box_point_test_demo()
