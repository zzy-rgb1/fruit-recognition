
# ###
# 本代码共采用了四种数据增强，如采用其他数据增强方式，可以参考本代码，随意替换。
# imageDir 为原数据集的存放位置
# saveDir  为数据增强后数据的存放位置
# ###

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def fl_ro_comb(root_path,img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20) #旋转角度
    comb_img = img.transpose(rotation_img.FLIP_LEFT_RIGHT)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return comb_img

def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    # 将图像的模式转换为正确的模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored

def combinationEnhancement(root_path,img_name): # 多种加强方式组合
    image = Image.open(os.path.join(root_path, img_name))
    # 颜色增强
    enh_color  = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_color.enhance(color)
    # 加强亮度
    enh_bri = ImageEnhance.Brightness(image_colored)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = 1.5
    image_com = enh_con.enhance(contrast)

    return image_com


from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np

dataset_path = 'fruit81_full'
dataset_name = dataset_path.split('_')[0]   # 取出文件名的前一半fruit81
dataset_enhanced_name = dataset_name + '_enhanced'
classes = os.listdir(dataset_path)      # 获取所有类别名称

# 对训练集的数据进行数据增强

# 创建保存数据加强后图片的文件夹
os.mkdir(dataset_enhanced_name)

for fruit in classes:
    # 读取该类别的所有图像文件名
    imageDir = os.path.join(dataset_path, fruit)

    # 创建各类别子文件夹
    os.mkdir(os.path.join(dataset_enhanced_name, fruit))

    for name in os.listdir(imageDir):
        # 对数据集的图片进行批量重命名，并保存到另一个目录中
        # image = Image.open(os.path.join(imageDir, name))
        # image.save(os.path.join(saveDir,saveName))

        saveName = name.split('.')[0] + "be.png"   # 给增强亮度后的图像设定名字
        saveImage=brightnessEnhancement(imageDir,name)
        enhanced_path = os.path.join(dataset_enhanced_name, fruit)  # 获取目录的新文件路径
        # rgbImage = saveImage.convert('RGB')
        saveImage.save(os.path.join(enhanced_path,saveName),'PNG')

        # saveName = name.split('.')[0] + "fl.png"   # 翻转图片
        # saveImage=flip(imageDir,name)
        # enhanced_path = os.path.join('fruit81_enhanced', fruit)  # 获取目录的新文件路径
        # rgbImage = saveImage.convert('RGB')
        # rgbImage.save(os.path.join(enhanced_path,saveName),'PNG')

        # saveName = name.split('.')[0] + "ro_fl.png"   # 旋转+翻转图片
        # saveImage = fl_ro_comb(imageDir, name)
        # enhanced_path = os.path.join('fruit81_enhanced', fruit)  # 获取目录的新文件路径
        # rgbImage = saveImage.convert('RGB')
        # rgbImage.save(os.path.join(enhanced_path,saveName),'PNG')

        # # 组合多种方式
        # saveName = name.split('.')[0] + "comb.png"   # 颜色、亮度和对比度加强的图片
        # saveImage = combinationEnhancement(imageDir, name)
        # enhanced_path = os.path.join('fruit81_enhanced', fruit)  # 获取目录的新文件路径
        # rgbImage = saveImage.convert('RGB')
        # rgbImage.save(os.path.join(enhanced_path,saveName),'PNG')

    print("完成对{}文件夹的数据加强".format(fruit))
