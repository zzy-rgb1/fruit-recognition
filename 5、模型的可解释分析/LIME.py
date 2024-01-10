import os

# 存放测试图片
os.mkdir('test_img')

# 存放模型权重文件
os.mkdir('checkpoint')

import lime
import sklearn

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

#加载被测试图片与模型
img_path = 'F:/WORKS/Jupyter/Train_Custom_Dataset-main/图像分类/test_img/106.jpg'
img_pil = Image.open(img_path)
model = torch.load('F:/WORKS/Jupyter/Train_Custom_Dataset-main/图像分类/checkpoint/fruit30_pytorch_C1_me.pth')
model = model.eval().to(device)
idx_to_labels = np.load('F:/WORKS/Jupyter/Train_Custom_Dataset-main/图像分类/idx_to_labels.npy', allow_pickle=True).item()
len(idx_to_labels)


idx_to_labels

# 预处理
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    trans_norm
    ])

trans_B = transforms.Compose([
        transforms.ToTensor(),
        trans_norm
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])

# 原始输入图像分类预测
input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1)
top_n = pred_softmax.topk(5)


top_n

# 定义分类预测函数
def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


test_pred = batch_predict([trans_C(img_pil)])
test_pred.squeeze().argmax()

#LIME可解释性分析
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(trans_C(img_pil)),
                                         batch_predict, # 分类预测函数
                                         top_labels=len(idx_to_labels),
                                         hide_color=0,
                                         num_samples=3000) # LIME生成的邻域图像个数


explanation.top_labels[0]

# 可视化
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
img_boundry = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry)
plt.show()