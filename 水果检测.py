from cProfile import label
from tensorflow.keras.layers import Dense
import tensorflow as tf
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sympy.plotting.textplot import rescale
from tensorflow.keras import models,layers
import os
import shutil
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.data_utils import validate_file
# from 手搓图像分类 import input_shape, num_class, batch_size, early_stop

# 设置路径 设置水果图片的位置
source_dir = {
    "apple": r"C:\fruits\apple",
    "banana":r"C:\fruits\banana",
    "grape":r"C:\fruits\grape",
    "orange":r"C:\fruits\orange",
    "pear":r"C:\fruits\pear"

}

# 设置训练集 验证集 测试集的位置
target_dir = {
    "test":r"C:\fruit train\test",
    "train":r"C:\fruit train\train",
    "val":r"C:\fruit train\val"

}


# 创建目标文件夹结构
for split in ["train","val","test"]:
    for class_name in ["apple","banana","grape","orange","pear"]:
        os.makedirs(os.path.join(target_dir[split],class_name),exist_ok=True)

# 这行主要是通过target_dir 拼接 split 然后拼接class name 然后exist = True

# 设置划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
# 训练集和验证集和测试集遵循7，2，1的划分比例
print(f"\n划分数据集")
print(f"训练集{train_ratio * 100}% ,验证集{val_ratio * 100}% ,测试集{test_ratio * 100}%")
# 对每个类别分别进行划分
for class_name,source_dir in source_dir.items():
    all_images = [f for f in os.listdir(source_dir)
    if f.lower().endswith((".png",".jpg",".bmp"))]
    print(f"\n{class_name}类别：共{len(all_images)}张照片")

# 第一次划分: 分离出训练集
    train_files,temp_files = train_test_split(
        all_images,
        train_size=train_ratio,
        random_state = 42,
        shuffle=True
    )
# 第二次划分: 从剩余文件中分离验证集和测试集
    val_file,test_file = train_test_split(
        temp_files,
        train_size = val_ratio / (val_ratio+test_ratio),
        random_state = 42,
        shuffle=True
    )

# 复制文件到对应目录
    for f in train_files:
        src = os.path.join(source_dir,f)
        dst = os.path.join(target_dir["train"],class_name,f)
        shutil.copy2(src,dst)

    for f in val_file:
        src = os.path.join(source_dir,f)
        dst = os.path.join(target_dir["val"],class_name,f)
        shutil.copy2(src,dst)

    for f in test_file:
        src = os.path.join(source_dir,f)
        dst = os.path.join(target_dir["test"],class_name,f)
        shutil.copy2(src,dst)

# 图像参数
img_height,img_width= 255,255
input_shape = (img_height,img_width,3)
num_class = 5
batch_size = 16

# 创建数据生成器 直接在数据生成归一，不在模型归一

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # 随机旋转±20度
    width_shift_range=0.2,   # 随机水平偏移±20%
    height_shift_range=0.2,  # 随机垂直偏移±20%
    horizontal_flip=True,    # 随机水平翻转
    zoom_range=0.2,          # 随机缩放±20%
    shear_range=0.2          # 随机剪切变换
)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 经典归一化


# 创建数据流
train_generator = train_datagen.flow_from_directory(
    target_dir["train"],
    target_size = (img_height,img_width),
    batch_size=batch_size,
    class_mode = "sparse"
)

val_generator = train_datagen.flow_from_directory(
    target_dir["val"],
    target_size = (img_height,img_width),
    batch_size=batch_size,
    class_mode = "sparse"

)
test_generator = train_datagen.flow_from_directory(
    target_dir["test"],
    target_size = (img_height,img_width),
    batch_size=batch_size,
    class_mode = "sparse"
)

print(f"训练集：{train_generator.samples}张图片")
print(f"验证集：{val_generator.samples}张图片")
print(f"测试集:{test_generator.samples}张图片")
# 构建CNN模型
model = models.Sequential()

# 卷积块1
model.add(layers.Conv2D(32,(3,3),activation = "relu", input_shape=input_shape)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))

# 卷积块2
model.add(layers.Conv2D(64,(3,3),activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))
# 全连接层
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128,activation = "relu"))
model.add(layers.BatchNormalization())
model.add(Dense(num_class,activation = "softmax"))

# 打印模型结构
print(model.summary())

# 编译模型
model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 添加早停回调
early_stop = EarlyStopping(
    monitor= "val_accuracy",
    patience = 5,
    restore_best_weights = True
)

# 训练模型
print("开始训练")
history = model.fit(
    train_generator,
    epochs = 20,
    validation_data = val_generator,
    callbacks = [early_stop]
)

# plt画图
plt.plot(history.history["accuracy"],label = "Training Accuracy")
plt.plot(history.history["val_accuracy"],label = "validation Accuracy")
plt.legend()  # 创建图列
plt.show()

# 评估模型
print("开始评估模型")
test_loss,test_acc = model.evaluate(test_generator)
# 保存模型
model.save("fruit mode.h5")
print("\n已经将模型保存为：fruit mode.h5")