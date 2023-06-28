from flask import Flask, request, send_file
from model import *
import os
import torch
import SimpleITK as sitk

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

app = Flask(__name__)

# 载入模型
newSize = (112, 112, 128)
Unet3d = MutilUNet3dModel(image_depth=128, image_height=112, image_width=112, image_channel=1, numclass=1,
                          batch_size=1, loss_name='MutilFocalLoss', inference=True,
                          model_path=r'log\MutilUNet3d\focalloss\BinaryVNet2dSegModel.pth')

root_dir = r"D:/uploads/Image"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

root_Mask_dir = r"D:/uploads/Mask"
if not os.path.exists(root_Mask_dir):
    os.makedirs(root_Mask_dir)


# 定义服务接口
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')  # 获取上传的文件
    if file:
        file.save(root_dir + '/' + file.filename)  # 将上传文件保存到本地
        sitk_image = sitk.ReadImage(root_dir + '/' + file.filename)  # 读取本地文件
        sitk_mask = Unet3d.inference(sitk_image, newSize)  # 对本地文件进行推理计算
        # 返回预测结果
        sitk.WriteImage(sitk_mask, root_Mask_dir + '/' + file.filename)
        return 'Segmentation Success!'
    else:
        return 'No file uploaded'


# 定义服务接口
@app.route('/getresult', methods=['GET'])
def getresult():
    filename = request.args.get('file')  # 获取请求参数中的文件名
    if not filename:
        return "Missing parameter: file"  # 没有提供文件名
    filepath = root_Mask_dir + '/' + filename  # 生成完整的文件路径
    try:
        return send_file(filepath, as_attachment=True, attachment_filename=filename)
    except FileNotFoundError:
        return "The file does not exist"  # 文件不存在


if __name__ == '__main__':
    # 使用curl命令行工具发送基于文件的POST请求来测试服务，192.168.10.96是服务器的ip
    # curl -X POST -F "file=@E:/TRAIN000106.nii.gz" 192.168.10.96:8000/predict
    # 使用curl命令行工具发送基于文件的GET请求来测试服务
    # curl 192.168.10.96:8000/getresult?file=TRAIN000106.nii.gz -o /home/Project/TRAIN000106.nii.gz
    app.run(host='0.0.0.0', port=8000)
