# 设置编码为UTF-8
# coding: utf-8

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import classification_report
from torchvision import datasets
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

matplotlib.rc("font", family='SimHei')  # 中文字体

n = 3
iters = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])

# 数据集文件夹路径
dataset_dir = './fruit81_full_split'
test_path = os.path.join(dataset_dir, 'test')

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)
model = torch.load('fruit81_pytorch_C1.pth')
model = model.eval().to(device)
img_paths = [each[0] for each in test_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets

df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]

df_pred = pd.DataFrame()
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    pred_dict = {}

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别

    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i - 1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()

    df_pred = df_pred.append(pred_dict, ignore_index=True)
    print(iters)
    iters += 1

df = pd.concat([df, df_pred], axis=1)
df.to_csv('测试集预测结果.csv', index=False)

df = pd.read_csv('测试集预测结果.csv')

print("预测准确率", sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df))

report = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()

df_report.to_csv('各类别准确率评估指标.csv', index_label='类别')

wrong_df = df[(df['标注类别名称']) != (df['top-1-预测名称'])]


i = 0
for idx, row in wrong_df.iterrows():
    img_path = row['图像路径']


    # img_bgr = cv2.imread(img_path)

    def readImg(path, mode):
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, mode)
        return img


    img_bgr = readImg(img_path, -1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    title_str = img_path + '\nTrue:' + row['标注类别名称'] + ' Pred:' + row['top-1-预测名称']
    plt.title(title_str)

    figure_save_path = "wrong_df"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建

    plt.savefig(os.path.join(figure_save_path, str(i) + '.jpg'))  # 第一个是指存储路径，第二个是图片名字
    i += 1
    print(i)
    # plt.show()

report = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('各类别准确率评估指标.csv', index_label='类别')

df = pd.read_csv('各类别准确率评估指标.csv')

feature = 'f1-score'
df_plot = df.sort_values(by=feature, ascending=False)

plt.figure(figsize=(40, 10))

x = df_plot['类别']
y = df_plot[feature]

ax = plt.bar(x, y, width=0.6, facecolor='#1f77b4', edgecolor='k')
plt.bar_label(ax, fmt='%.2f', fontsize=7) # 置信度数值

plt.xticks(rotation=45)
plt.tick_params(labelsize=7)
plt.xlabel('类别', fontsize=20)
plt.ylabel(feature, fontsize=15)
plt.title('准确率评估指标 {}'.format(feature), fontsize=25)

plt.savefig('各类别准确率评估指标柱状图-{}.pdf'.format(feature), dpi=120, bbox_inches='tight')

plt.show()

df = pd.read_csv('测试集预测结果.csv')
from matplotlib import colors as mcolors
import random

random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick',
          'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen',
          'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray',
          'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue',
          'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid',
          'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred',
          'deeppink', 'hotpink']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
linestyle = ['--', '-.', '-']


def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.plot([0, 1], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.rcParams['font.size'] = 22
plt.grid(True)

auc_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
    plt.legend()
    auc_list.append(auc(fpr, tpr))

plt.legend(loc='best', fontsize=12)
plt.savefig('各类别ROC曲线.pdf'.format(classes), dpi=120, bbox_inches='tight')
plt.show()

df_report = pd.read_csv('各类别准确率评估指标.csv')
