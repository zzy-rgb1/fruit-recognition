import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt

# 指定数据集路径
dataset_path = 'fruit81_full'
os.chdir(dataset_path)
os.listdir()

df = pd.DataFrame()
for fruit in tqdm(os.listdir()): # 遍历每个类别
    os.chdir(fruit)
    for file in os.listdir(): # 遍历每张图像
        try:
            img = cv2.imread(file)
            df = df._append({'类别':fruit, '文件名':file, '图像宽':img.shape[1], '图像高':img.shape[0]}, ignore_index=True)
        except:
            print(os.path.join(fruit, file), '读取错误')
    os.chdir('../')
os.chdir('../')

# 可视化图像尺寸分布
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

x = df['图像宽']
y = df['图像高']

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# 创建一个散点图，并根据数据点的密度值对散点进行着色和调整点的大小，以显示出数据点的分布情况和密度差异
plt.figure(figsize=(10,10))         # 创建新的图形窗口
plt.scatter(x, y, c=z,  s=5, cmap='Spectral_r')

plt.tick_params(labelsize=10)

xy_max = max(max(df['图像宽']), max(df['图像高']))
plt.xlim(xmin=0, xmax=xy_max)
plt.ylim(ymin=0, ymax=xy_max)

plt.ylabel('height', fontsize=20)
plt.xlabel('width', fontsize=20)

# 将当前图形保存为名为 '图像尺寸分布.pdf' 的 PDF 文件。dpi 参数指定图像的分辨率，bbox_inches='tight' 参数用于裁剪多余的空白区域。
plt.savefig('图像尺寸分布.pdf', dpi=120, bbox_inches='tight')
# 将其存为png的格式
# plt.savefig('图像尺寸分布.png', dpi=120, bbox_inches='tight')

plt.show()

