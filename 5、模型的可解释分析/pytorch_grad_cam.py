import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
%matplotlib inline

import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, GradCAMElementWise
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 载入模型
model = torch.load('checkpoint/fruit30_pytorch_C1_me.pth')
#model = torch.load('checkpoint/fruit30_pytorch_20220814.pth')
#model = torch.load('checkpoint/fruit81_pytorch_C1_2.pth')
model = model.eval().to(device)

idx_to_labels_cn = np.load('idx_to_labels.npy', allow_pickle=True).item()
idx_to_labels_cn




# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

#载入测试图片
img_path = 'test_img/31.jpg'
img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
input_tensor.shape
# GradCAM 选择可解释性分析方法
from pytorch_grad_cam import GradCAM
target_layers = [model.layer4[-1]] # 要分析的层
targets = [ClassifierOutputTarget(78)] # 要分析的类别
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

#生成Grad-CAM热力图
cam_map = cam(input_tensor=input_tensor, targets=targets)[0] # 不加平滑
# cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0] # 加平滑

cam_map.shape

plt.imshow(cam_map)
plt.title('Grad-CAM')
plt.show()

import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.5) # alpha越小，原图越淡

plt.imshow(result)
plt.title('Grad-CAM')
plt.show()