import numpy as np
import time
from PIL import Image
import SimpleITK as sitk
import torch
from sam2.build_sam import build_sam2_video_predictor_npz


class tumorSAM3dwithMutilBoxPoint_Inference:
    def __init__(self, propagate_with_box=True, sample_points='from_box'):
        checkpoint = r"weight/MedSAM2_CTLesion.pt"
        model_cfg = r"configs/sam2.1_hiera_t512.yaml"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.propagate_with_box = propagate_with_box
        self.sample_points = sample_points
        # initialized predictor
        self.predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    def sample_points_in_bbox_grid(self, bbox: np.ndarray, n: int) -> np.ndarray:
        """
        Uniformly sample n grid-aligned (x, y) points inside the bbox.

        Args:
            bbox (np.ndarray): [x_min, y_min, x_max, y_max]
            n (int): Number of points to sample

        Returns:
            np.ndarray: shape (n, 2), each row is [x, y]
        """
        x_min, y_min, x_max, y_max = bbox
        grid_size = int(np.ceil(np.sqrt(n)))

        x_vals = np.linspace(x_min, x_max, grid_size, dtype=int)
        y_vals = np.linspace(y_min, y_max, grid_size, dtype=int)

        xv, yv = np.meshgrid(x_vals, y_vals)
        coords = np.stack([xv.ravel(), yv.ravel()], axis=1)

        return coords[:n]

    def resize_grayscale_to_rgb_and_resize(self, array, image_size):
        """
        Resize a 3D grayscale NumPy array to an RGB image and then resize it.

        Parameters:
            array (np.ndarray): Input array of shape (d, h, w).
            image_size (int): Desired size for the width and height.

        Returns:
            np.ndarray: Resized array of shape (d, 3, image_size, image_size).
        """
        d, h, w = array.shape
        resized_array = np.zeros((d, 3, image_size, image_size))

        for i in range(d):
            img_pil = Image.fromarray(array[i].astype(np.uint8))
            img_rgb = img_pil.convert("RGB")
            img_resized = img_rgb.resize((image_size, image_size))
            img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
            resized_array[i] = img_array

        return resized_array

    def medsam2_seg3d_infer(self, img_3D_ori, unique_labs_list):
        """
        :param img_3D_ori:imge3d
        :param unique_labs_list:[[x1,y1,z1,x2,y2,x2,label],[x1,y1,z1,x2,y2,x2,label]]
        :return:
        """
        segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
        # resize image to 512x512 and normalize
        video_height, video_width = img_3D_ori.shape[1:3]
        if video_height != 512 or video_width != 512:
            img_resized = self.resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)  # 1024) #d, 3, 1024, 1024
        else:
            img_resized = img_3D_ori[:, None].repeat(3, axis=1)  # d, 3, 1024, 1024
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
        img_resized -= img_mean
        img_resized /= img_std

        # iterate over each label
        for j in range(len(unique_labs_list)):
            unique_lab = unique_labs_list[j]  # y_min, x_min,z_min, y_max, x_max,z_max,labelvalue
            z_min = unique_lab[2]
            z_max = unique_lab[5]
            idx = unique_lab[-1]
            z_list = [z_min, z_max]
            z_list = list(set(z_list))
            # add prompt to initialize the predictor
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # iterate over each 3dbox
                for z in range(len(z_list)):
                    key_slice_idx_offset = z_list[z]
                    inference_state = self.predictor.init_state(img_resized, video_height, video_width)
                    if self.propagate_with_box:
                        print('propagate with box')
                        box_2d = np.array(
                            [unique_lab[0], unique_lab[1], unique_lab[3], unique_lab[4]])  # y_min, x_min, y_max, x_max
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=box_2d,
                        )
                        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                    else:
                        print('propagate with point')
                        if self.sample_points == 'from_box':
                            box_2d = np.array([unique_lab[0], unique_lab[1], unique_lab[3], unique_lab[4]])
                            # sample points in the box
                            points = self.sample_points_in_bbox_grid(box_2d, n=5)
                        else:
                            box_2d = np.array([unique_lab[0], unique_lab[1], unique_lab[3], unique_lab[4]])
                            x_min, y_min, x_max, y_max = box_2d
                            points = np.array([[x_min, y_min], [x_max, y_min],
                                               [x_max, y_max], [x_min, y_max]], dtype=int)
                        labels = np.ones(len(points))  # all positive points
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points,
                            labels=labels,
                        )
                        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

                    # start inference propagation
                    frame_idx, object_ids, masks = self.predictor.add_new_mask(inference_state,
                                                                               frame_idx=key_slice_idx_offset,
                                                                               obj_id=1, mask=mask_prompt)
                    segs_3D[key_slice_idx_offset, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=key_slice_idx_offset,
                            reverse=False):
                        segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
                    # reverse process, delete old memory and initialize new predictor
                    self.predictor.reset_state(inference_state)
                    inference_state = self.predictor.init_state(img_resized, video_height, video_width)
                    frame_idx, object_ids, masks = self.predictor.add_new_mask(inference_state,
                                                                               frame_idx=key_slice_idx_offset,
                                                                               obj_id=1,
                                                                               mask=mask_prompt)
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=key_slice_idx_offset,
                            reverse=True):
                        segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
                    self.predictor.reset_state(inference_state)
        return segs_3D

    def network_prediction(self, inputfilepath, unique_labs_list, lower_bound=-100, upper_bound=200):
        """
        :param inputfilepath: image path
        :param unique_labs_list: [[x1,y1,z1,x2,y2,z2,label],[x1,y1,z1,x2,y2,z2,label],[x1,y1,z1,x2,y2,z2,label]]
        :return:
        """
        if not (inputfilepath.endswith('.nii') or inputfilepath.endswith('.nii.gz') or inputfilepath.endswith('.mha')):
            print("文件格式不支持，仅支持 .nii, .nii.gz 和 .mha 格式")
            return False, None
        try:
            nii_image = sitk.ReadImage(inputfilepath)
            nii_image_data = sitk.GetArrayFromImage(nii_image)
            lower_bound, upper_bound = float(lower_bound), float(upper_bound)
            nii_image_data_pre = np.clip(nii_image_data, lower_bound, upper_bound)
            nii_image_data_pre = (nii_image_data_pre - np.min(nii_image_data_pre)) / (
                    np.max(nii_image_data_pre) - np.min(nii_image_data_pre)) * 255.0
            img_3D_ori = np.uint8(nii_image_data_pre)
            assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'
            array_mask = self.medsam2_seg3d_infer(img_3D_ori, unique_labs_list)
            sitk_mask = sitk.GetImageFromArray(array_mask.astype('uint8'))
            sitk_mask.CopyInformation(nii_image)
            return True, sitk_mask
        except Exception as e:
            print(f"出现异常：{e}", inputfilepath)
            return False, None


if __name__ == '__main__':
    input_image_path = r"C:\liver_image.nii.gz"
    output_mask_path = "liver_tumor_SAM3d_point.nii.gz"
    box_mask_path = "SAM3d_box3d.nii.gz"
    # boundingbox
    # box_list_2d_liver_tumor = [135, 344, 382, 151, 364, 382, 4]  # x1,y1,z1,x2,y2,z2,label
    # box_list_2d_kidney_tumor = [154, 347, 277, 167, 360, 277, 2]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_liver_tumor = [141, 354, 383, 144, 360, 383, 4]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_kidney_tumor = [158, 354, 275, 161, 357, 275, 2]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_vessel = [263, 252, 360, 315, 305, 360, 3]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_liver = [78, 141, 380, 280, 380, 380, 1]  # x1,y1,z1,x2,y2,z2,label
    box_list_3d_list = [box_list_2d_liver_tumor, box_list_2d_kidney_tumor]
    start = time.time()
    tumorsam3d = tumorSAM3dwithMutilBoxPoint_Inference(propagate_with_box=False, sample_points='pointself')
    _, sitk_mask = tumorsam3d.network_prediction(input_image_path, box_list_3d_list)
    end = time.time()
    print(end - start)
    sitk.WriteImage(sitk_mask, output_mask_path)

    box_mask = np.zeros_like(sitk.GetArrayFromImage(sitk_mask))
    box_mask[box_list_2d_liver[2], box_list_2d_liver[1]:box_list_2d_liver[4],
    box_list_2d_liver[0]:box_list_2d_liver[3]] = box_list_2d_liver[-1]
    box_mask[box_list_2d_liver_tumor[2], box_list_2d_liver_tumor[1]:box_list_2d_liver_tumor[4],
    box_list_2d_liver_tumor[0]:box_list_2d_liver_tumor[3]] = box_list_2d_liver_tumor[-1]
    box_mask[box_list_2d_kidney_tumor[2], box_list_2d_kidney_tumor[1]:box_list_2d_kidney_tumor[4],
    box_list_2d_kidney_tumor[0]:box_list_2d_kidney_tumor[3]] = box_list_2d_kidney_tumor[-1]
    box_mask[box_list_2d_vessel[2], box_list_2d_vessel[1]:box_list_2d_vessel[4],
    box_list_2d_vessel[0]:box_list_2d_vessel[3]] = box_list_2d_vessel[-1]
    box_mask_sitk = sitk.GetImageFromArray(box_mask)
    box_mask_sitk.CopyInformation(sitk_mask)
    sitk.WriteImage(box_mask_sitk, box_mask_path)
