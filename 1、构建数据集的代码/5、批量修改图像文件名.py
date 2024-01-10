import os

from PIL import Image

dataset_path = 'fruit81_enhanced_split'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
classes = os.listdir(train_path)      # 获取所有类别名称

# 修改训练集中图片的名字
for fruit in classes:
    fruit_dir_train = os.path.join(train_path, fruit)
    fruit_dir_val = os.path.join(val_path, fruit)
    images_filename_train = os.listdir(fruit_dir_train)    # 对应训练集一个种类下的所有图片名字
    images_filename_val = os.listdir(fruit_dir_val)    # 对应测试集一个种类下的所有图片名字
    for index, image_name in enumerate(images_filename_train):
        image_path_old = os.path.join(fruit_dir_train, image_name)
        saveName = str(index+300) + '.' + image_name.split('.')[1]   # 修改图片名字
        image_path_new = os.path.join(fruit_dir_train, saveName)
        os.rename(image_path_old, image_path_new)
    print("train中的{}水果文件重命名成功！".format(fruit))
        # image = Image.open(image_path)
        # saveName = str(index) + '.' + image_name.split('.')[1]   # 修改图片名字
        # image.save(os.path.join(fruit_dir_train, saveName))
    for index, image_name in enumerate(images_filename_val):
        image_path_old = os.path.join(fruit_dir_val, image_name)
        saveName = str(index+300) + '.' + image_name.split('.')[1]   # 修改图片名字
        image_path_new = os.path.join(fruit_dir_val, saveName)
        os.rename(image_path_old, image_path_new)
        # image.save(os.path.join(fruit_dir_val, saveName))
    print("val中的{}水果文件重命名成功！".format(fruit))
