import os
import shutil
import random
import pandas as pd

# 获得所有类别名称
# 指定数据集路径
dataset_path = 'fruit81_enhanced'
dataset_name = dataset_path.split('_')[0]   # 取出文件名的前一半fruit81
print('数据集', dataset_name)
classes = os.listdir(dataset_path)      # 获取所有类别名称

# 创建训练集文件夹和测试集文件夹
# 创建 train 文件夹
os.mkdir(os.path.join(dataset_path, 'train'))  # fruit_train

# 创建 test 文件夹
os.mkdir(os.path.join(dataset_path, 'val'))     # fruit_test

# 在 train 和 test 文件夹中创建各类别子文件夹``
for fruit in classes:
    os.mkdir(os.path.join(dataset_path, 'train', fruit))
    os.mkdir(os.path.join(dataset_path, 'val', fruit))

# 划分训练集、测试集，移动文件
test_frac = 0.2  # 测试集比例
random.seed(123) # 随机数种子，便于复现

# 展示数据用的，类似于一个二维表格，由行和列组成，每一列可以包含不同类型的数据（整数、浮点数、字符串等），而每一行则表示数据的一个观测点或样本。
df = pd.DataFrame()

print('{:^18} {:^18} {:^18}'.format('类别', '训练集数据个数', '测试集数据个数'))

for fruit in classes:  # 遍历每个类别

    # 读取该类别的所有图像文件名
    old_dir = os.path.join(dataset_path, fruit)
    images_filename = os.listdir(old_dir)       # 某种类别下的图像文件名字
    random.shuffle(images_filename)  # 随机打乱

    # 划分训练集和测试集
    testset_numer = int(len(images_filename) * test_frac)  # 测试集图像个数
    testset_images = images_filename[:testset_numer]  # 获取拟移动至 test 目录的测试集图像文件名
    trainset_images = images_filename[testset_numer:]  # 获取拟移动至 train 目录的训练集图像文件名

    # 移动图像至 test 目录
    for image in testset_images:
        old_img_path = os.path.join(dataset_path, fruit, image)  # 获取原始文件路径
        new_test_path = os.path.join(dataset_path, 'val', fruit, image)  # 获取 test 目录的新文件路径
        shutil.move(old_img_path, new_test_path)  # 移动文件

    # 移动图像至 train 目录
    for image in trainset_images:
        old_img_path = os.path.join(dataset_path, fruit, image)  # 获取原始文件路径
        new_train_path = os.path.join(dataset_path, 'train', fruit, image)  # 获取 train 目录的新文件路径
        shutil.move(old_img_path, new_train_path)  # 移动文件

    # 删除旧文件夹
    assert len(os.listdir(old_dir)) == 0  # 确保旧文件夹中的所有图像都被移动走
    shutil.rmtree(old_dir)  # 删除文件夹

    # 工整地输出每一类别的数据个数
    print('{:^18} {:^18} {:^18}'.format(fruit, len(trainset_images), len(testset_images)))

    # 保存到表格中
    df = df._append({'class': fruit, 'trainset': len(trainset_images), 'testset': len(testset_images)},
                   ignore_index=True)

# 重命名数据集文件夹
shutil.move(dataset_path, dataset_name + '_enhanced_split')

# 数据集各类别数量统计表格，导出为 csv 文件
df['total'] = df['trainset'] + df['testset']    # 新增加一个名为total的新列，该列包含了trainset和testset列对应位置的值相加的结果
# '数据量统计.csv'是保存的文件路径，index=False表示不保存行索引。
df.to_csv('数据量统计.csv', index=False)

