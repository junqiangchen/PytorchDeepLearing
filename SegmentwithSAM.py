# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import transform
import time


class tumorSAM_Inference:
    def __init__(self):
        checkpoint = r"weight/medsam_vit_b.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.medsam_model = self.medsam_model.to(self.device)
        self.medsam_model.eval()

    @torch.no_grad()
    def medsam_inference(self, medsam_model, img_embed, box_1024, H, W):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def medsam_seg2d_infer(self, img_np, box_list2d):
        """
        :param img_np:imge2d
        :param box_list2d:[x1,y1,x2,y2]
        :return:
        """
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        # %% image preprocessing
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3,
                                    preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(),
                                                         a_min=1e-8, a_max=None)  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device))
        box_np = np.array([box_list2d])
        # transfer box_np t0 1024x1024 scale
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            image_embedding = self.medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        medsam_seg = self.medsam_inference(self.medsam_model, image_embedding, box_1024, H, W)
        return medsam_seg

    def network_prediction(self, inputfilepath, box_list_3d):
        """
        :param inputfilepath: image path
        :param box_list_3d: [x1,y1,z1,x2,y2,z2]
        :return:
        """
        if not (inputfilepath.endswith('.nii') or inputfilepath.endswith('.nii.gz') or inputfilepath.endswith('.mha')):
            print("文件格式不支持，仅支持 .nii, .nii.gz 和 .mha 格式")
            return False, None
        try:
            sitk_image = sitk.ReadImage(inputfilepath)
            array_image = sitk.GetArrayFromImage(sitk_image)
            x1, y1, z1, x2, y2, z2 = (box_list_3d[0], box_list_3d[1], box_list_3d[2],
                                      box_list_3d[3], box_list_3d[4], box_list_3d[5])
            array_mask = np.zeros_like(array_image)
            for i in range(z1, z2, 1):
                img_np = array_image[i, :, :]  # 选择某一层切片
                box_list = [x1, y1, x2, y2]  # [x1, y1, x2, y2]
                medsam_seg2d = self.medsam_seg2d_infer(img_np, box_list)
                array_mask[i, :, :] = medsam_seg2d.copy()
            sitk_mask = sitk.GetImageFromArray(array_mask.astype('uint8'))
            sitk_mask.CopyInformation(sitk_image)
            return True, sitk_mask
        except Exception as e:
            print(f"出现异常：{e}", inputfilepath)
            return False, None


if __name__ == '__main__':
    input_image_path = r"C:\liver_image.nii.gz"
    output_mask_path = "liver_tumor_SAM.nii.gz"
    box_mask_path = "SAM_box3d.nii.gz"
    box_list_3d = [135, 344, 377, 151, 364, 388]  # x1,y1,z1,x2,y2,z2
    start = time.time()
    tumorsam2d = tumorSAM_Inference()
    _, sitk_mask = tumorsam2d.network_prediction(input_image_path, box_list_3d)
    end = time.time()
    print(end - start)
    sitk.WriteImage(sitk_mask, output_mask_path)

    box_mask = np.zeros_like(sitk.GetArrayFromImage(sitk_mask))
    box_mask[box_list_3d[2]:box_list_3d[5], box_list_3d[1]:box_list_3d[4], box_list_3d[0]:box_list_3d[3]] = 1
    box_mask_sitk = sitk.GetImageFromArray(box_mask)
    box_mask_sitk.CopyInformation(sitk_mask)
    sitk.WriteImage(box_mask_sitk, box_mask_path)
