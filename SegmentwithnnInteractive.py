import os

import numpy as np
import torch
import SimpleITK as sitk
import time
from collections import defaultdict
# --- Initialize Inference Session ---
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


class tumornnInteractivewithMutilBoxPoint_Inference:
    """
    project:https://github.com/MIC-DKFZ/nnInteractive?tab=readme-ov-file,
    model:https://huggingface.co/nnInteractive/nnInteractive/tree/main
    """
    def __init__(self, propagate_with_type='box'):
        # propagate_with_type support point,box,mask,lasso
        self.session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        # Load the trained model
        model_path = r"weight\nnInteractive_v1.0"
        self.session.initialize_from_trained_model_folder(model_path)
        self.propagate_with_type = propagate_with_type

    def _seg3d_infer_withpointboxmasklasso(self, input_image, points_pos_list=None, points_neg_list=None,
                                           bboxs_list=None, mask_binary=None):
        # --- Load Input Image (Example with SimpleITK) ---
        # DO NOT preprocess the image in any way. Give it to nnInteractive as it is! DO NOT apply level window, DO NOT normalize
        # intensities and never ever convert an image with higher precision (float32, uint16, etc) to uint8!
        # The ONLY instance where some preprocesing makes sense is if your original image is too large to be reasonably used.
        # This may be the case, for example, for some microCT images. In this case you can consider downsampling.
        img = sitk.GetArrayFromImage(input_image)[None]  # Ensure shape (1, x, y, z)
        # Validate input dimensions
        if img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")
        self.session.set_image(img)
        # --- Define Output Buffer ---
        target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(target_tensor)
        # --- Interacting with the Model ---
        # Interactions can be freely chained and mixed in any order. Each interaction refines the segmentation.
        # The model updates the segmentation mask in the target buffer after every interaction.
        # Example: Add a **positive** point interaction
        # POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
        if self.propagate_with_type == 'point':
            if points_pos_list is not None:
                for i in range(len(points_pos_list)):
                    points = points_pos_list[i]
                    POINT_COORDINATEone = (points[2], points[1], points[0])
                    POINT_COORDINATEtwo = (points[5], points[4], points[3])
                    print(POINT_COORDINATEone)
                    print(POINT_COORDINATEtwo)
                    self.session.add_point_interaction(POINT_COORDINATEone, include_interaction=True)
                    self.session.add_point_interaction(POINT_COORDINATEtwo, include_interaction=True)
            if points_neg_list is not None:
                for i in range(len(points_neg_list)):
                    # Example: Add a **negative** point interaction
                    # To make any interaction negative set include_interaction=False
                    one_point = points_neg_list[i]
                    POINT_COORDINATES = (one_point[2], one_point[1], one_point[0])
                    print(POINT_COORDINATES)
                    self.session.add_point_interaction(POINT_COORDINATES, include_interaction=False)
        if self.propagate_with_type == 'box':
            # Example: Add a bounding box interaction
            # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
            # Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
            # This means that **one dimension must be [d, d+1]** to indicate a single slice.
            # Example of a 2D bounding box in the axial plane (XY slice at depth Z)
            # BBOX_COORDINATES = [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10
            if bboxs_list is not None:
                for i in range(len(bboxs_list)):
                    one_box = bboxs_list[i]
                    BBOX_COORDINATES = [[one_box[2], one_box[5] + 1], [one_box[1], one_box[4]],
                                        [one_box[0], one_box[3]]]
                    print(BBOX_COORDINATES)
                    self.session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)
        if self.propagate_with_type == 'mask':
            # Example: Add a scribble interaction
            # - A 3D image of the same shape as img where one slice (any axis-aligned orientation) contains a hand-drawn scribble.
            # - Background must be 0, and scribble must be 1.
            # - Use session.preferred_scribble_thickness for optimal results.
            mask_binary = mask_binary.astype('uint8')
            mask_binary[mask_binary != 0] = 1
            self.session.add_scribble_interaction(mask_binary, include_interaction=True)
        if self.propagate_with_type == 'lasso':
            # Example: Add a lasso interaction
            # - Similarly to scribble a 3D image with a single slice containing a **closed contour** representing the selection.
            mask_binary = mask_binary.astype('uint8')
            mask_binary[mask_binary != 0] = 1
            self.session.add_lasso_interaction(mask_binary, include_interaction=True)
        # You can combine any number of interactions as needed.
        # The model refines the segmentation result incrementally with each new interaction.
        # --- Retrieve Results ---
        # The target buffer holds the segmentation result.
        # results = self.session.target_buffer.clone()
        # OR (equivalent)
        results = target_tensor.clone()
        # Cloning is required because the buffer will be **reused** for the next object.
        # Alternatively, set a new target buffer for each object:
        self.session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))
        # --- Start a New Object Segmentation ---
        self.session.reset_interactions()  # Clears the target buffer and resets interactions
        return results

    def network_prediction(self, inputfilepath, unique_labs_list=None, sitk_mask_binary=None):
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
            array_mask = np.zeros_like(sitk.GetArrayFromImage(nii_image))
            if unique_labs_list is not None:
                # 先把 unique_labs_list 按照 label 分组，变成字典：
                grouped_boxes_dict = defaultdict(list)
                for box in unique_labs_list:
                    x1, y1, z1, x2, y2, z2, label = box
                    grouped_boxes_dict[label].append([x1, y1, z1, x2, y2, z2])
                grouped_boxes_dict = dict(sorted(grouped_boxes_dict.items(), key=lambda x: x[0]))
                print(grouped_boxes_dict)
                if self.propagate_with_type == 'point':
                    points_neg_list = None
                    for label, bboxes in grouped_boxes_dict.items():
                        print(f"类别: {label}")
                        if label == 0:
                            points_neg_list = bboxes
                            continue
                        points_pos_list = bboxes
                        one_label_array_mask = self._seg3d_infer_withpointboxmasklasso(nii_image,
                                                                                       points_pos_list=points_pos_list,
                                                                                       points_neg_list=points_neg_list)
                        array_mask[one_label_array_mask != 0] = label
                if self.propagate_with_type == 'box':
                    for label, bboxes in grouped_boxes_dict.items():
                        print(f"类别: {label}")
                        one_label_array_mask = self._seg3d_infer_withpointboxmasklasso(nii_image, bboxs_list=bboxes)
                        array_mask[one_label_array_mask != 0] = label
            elif sitk_mask_binary is not None:
                mask_binary = sitk.GetArrayFromImage(sitk_mask_binary)
                one_label_array_mask = self._seg3d_infer_withpointboxmasklasso(nii_image, mask_binary=mask_binary)
                array_mask[one_label_array_mask != 0] = np.max(mask_binary)
            else:
                print('pleas check input')
                return False, None
            sitk_mask = sitk.GetImageFromArray(array_mask.astype('uint8'))
            sitk_mask.CopyInformation(nii_image)
            return True, sitk_mask
        except Exception as e:
            print(f"出现异常：{e}", inputfilepath)
            return False, None


def box_point_test_demo():
    input_image_path = r"D:\liver_image.nii.gz"
    output_mask_path = "liver_tumor_nnInteractive_point.nii.gz"
    box_list_2d_liver_tumor = [141, 354, 383, 144, 360, 383, 4]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_kidney_tumor = [158, 354, 275, 161, 357, 275, 2]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_vessel = [263, 252, 360, 315, 305, 360, 3]  # x1,y1,z1,x2,y2,z2,label
    box_list_2d_liver = [78, 141, 380, 280, 380, 380, 1]  # x1,y1,z1,x2,y2,z2,label
    box_list_3d_list = [box_list_2d_liver_tumor, box_list_2d_kidney_tumor]
    start = time.time()
    tumorsam3d = tumornnInteractivewithMutilBoxPoint_Inference(propagate_with_type='point')
    _, sitk_mask = tumorsam3d.network_prediction(input_image_path, unique_labs_list=box_list_3d_list)
    end = time.time()
    print(end - start)
    sitk.WriteImage(sitk_mask, output_mask_path)


if __name__ == '__main__':
    box_point_test_demo()
