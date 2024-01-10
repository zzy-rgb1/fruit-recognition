import os
import time
import shutil

import mmcv

import cv2
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = ImageFont.truetype('SimHei.ttf', 32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('checkpoint/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)

#图片预处理参数设置
test_transform = transforms.Compose([transforms.Resize(256),#将输入图像的大小调整为256x256像素
									 transforms.CenterCrop(224),#从图像中心裁剪出224x224像素的区域
									 transforms.ToTensor(),#将裁剪后的图像转换为PyTorch张量，这也会将图像的像素强度值从0-255变为0-1（归一化）
									 transforms.Normalize(
										 mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
									 ])#对图像进行标准化

#对于单张图片进行预测处理
def pred_single_frame( img_path,img_pil, isPicture):
	input_img = test_transform(img_pil).unsqueeze(0).to(device)#预处理后增加维度，将图片移入设备
	pred_logits = model(input_img)#将预处理后的图像输入到模型中，得到预测的logits（未归一化的预测）
	pred_softmax = F.softmax(pred_logits, dim=1)#使用softmax函数将logits转换为概率。softmax函数可以确保所有的输出概率都在0和1之间，并且所有概率之和为1

	n = 10
	top_n = torch.topk(pred_softmax, n)#输出前n个最大值和这些最大值对应的索引，输出形式为张量，两个张量存储在一个元组中。
	pred_ids = top_n[1].cpu().detach().numpy().squeeze()
	confs = top_n[0].cpu().detach().numpy().squeeze()#将前n个最大值和对应的索引从PyTorch张量转换为NumPy数组，并移除任何多余的维度

	#将获得的信息写入至测试图像上
	draw = ImageDraw.Draw(img_pil)
	for i in range(n):
		class_name = idx_to_labels[pred_ids[i]]#取出索引id根据字典查为真实水果名
		confidence = confs[i] * 100#概率*100，即改为百分比制
		text = '{:<15} {:>.4f}'.format(class_name, confidence)#生成一个字符串，其中包含类别名（左对齐，总宽度为15）和置信度（右对齐，小数点后有4位数字）
		draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))#在测试图像上绘制文本，即生成预测图

	# 若进行处理图像原文件是照片，则进行输出预测图+柱状图保存
	if 1 == isPicture:
		fig = plt.figure(figsize=(18, 6))
		# 将预测图作为fig图像的左子图并去除了坐标轴
		ax1 = plt.subplot(1, 2, 1)
		ax1.imshow(img_pil)
		ax1.axis('off')
		# 生成一个包含了所有类别和相关置信度的柱状图作为fig图像的右子图
		ax2 = plt.subplot(1, 2, 2)
		x = idx_to_labels.values()  # 条形图的x轴标签，即所有水果的种类名，并非前n个
		y = pred_softmax.cpu().detach().numpy()[0] * 100  # 条形图的高度，即对应的水果种类的置信度
		bars = ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
		plt.bar_label(bars, fmt='%.2f', fontsize=10)
		plt.title('{} 图像分类预测结果'.format(img_path), fontsize=30)
		plt.xlabel('类别', fontsize=20)
		plt.ylabel('置信度', fontsize=20)
		plt.ylim([0, 110])
		ax2.tick_params(labelsize=16)
		plt.xticks(rotation=90)
		plt.tight_layout()
		output_filename = 'output/预测图+柱状图_' + os.path.splitext(os.path.basename(img_path))[0] + '.jpg'
		fig.savefig(output_filename)
		plt.close(fig)

	# 控制台实时输出预测数据，方便实时观察模型预测结果
	pred_df = pd.DataFrame()
	for i in range(n):
		class_name = idx_to_labels[pred_ids[i]]
		label_idx = int(pred_ids[i])
		confidence = confs[i] * 100
		pred_df = pred_df._append({'Class': class_name, 'Class_ID': label_idx, 'Confidence(%)': confidence},
								 ignore_index=True)
	print(pred_df.to_string())
	return img_pil#返回预测图便于组帧回视频


# 文件夹路径
folder_path = 'test_img'
# 获取文件夹中的所有文件
files_in_folder = os.listdir(folder_path)
# 遍历每个文件
for file in files_in_folder:
	file_path = os.path.join(folder_path, file)
	# 检查文件是否为图像（这里只检查了.jpg和.png格式的图像）其实这种方式严格来说不太准确，因为文件后缀是可以修改的，但是数据集被人篡改文件名概率很小，故采用此方式
	if file.endswith('.jpg') or file.endswith('.png'):
		img_pil = Image.open(file_path)
		img_pil = pred_single_frame( file_path,img_pil, 1)
	# 检查文件是否为视频（这里只检查了.mp4格式的视频）
	elif file.endswith('.mp4'):
		# 创建临时文件夹，存放每帧结果
		temp_out_dir = time.strftime('%Y%m%d%H%M%S')
		os.mkdir(temp_out_dir)
		# 读入待预测视频
		imgs = mmcv.VideoReader(file_path)
		prog_bar = mmcv.ProgressBar(len(imgs))
		# 对视频逐帧处理
		for frame_id, img in enumerate(imgs):
			img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			img_pil = pred_single_frame(file_path,img_pil, 0)
			# 将处理后的该帧画面图像文件，保存至临时目录下
			cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
			prog_bar.update()  # 更新进度条

		# 把每一帧串成视频文件
		output_filename = 'output/预测视频_' + os.path.splitext(os.path.basename(file_path))[0] + '.mp4'
		mmcv.frames2video(temp_out_dir, output_filename, fps=imgs.fps, fourcc='mp4v')
		# 删除存放每帧画面的临时文件夹
		shutil.rmtree(temp_out_dir)
