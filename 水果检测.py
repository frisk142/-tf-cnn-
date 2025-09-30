# 第1步：导入所有需要的包 (Import)
# python
# # 提示：os, shutil, 数据划分, tensorflow, 模型层, 回调函数, 画图
import shutil

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.dtensor.python.config import num_clients
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import regularizers

# 新增：所有类别名称的列表（必须和source_dira的键一致）
CLASS_NAMES = ["apple", "banana", "grape", "orange", "pear"]

source_dir = {
    "apple":r"C:\fruits\apple",
    "banana":r"C:\fruits\banana",
    "grape":r"C:\fruits\grape",
    "orange":r"C:\fruits\orange",
    "pear":r"C:\fruits\pear"
}

taget_dir = {
    "train":r"C:\fruit train\train",
    "val":r"C:\fruit train\val",
    "test":r"C:\fruit train\test",

}
# 提示：定义两个字典
# # source_dirs: 键是类别名，值是原始图片路径
# # target_dirs: 键是'train','val','test'，值是目标文件夹路径

for split in ["train","val","test"]:
    for class_name in ["apple", "banana", "grape", "orange", "pear"]:
        os.makedirs(os.path.join(taget_dir[split],class_name),exist_ok=True)

train_traio = 0.7
val_traio = 0.2
test_traio = 0.1
# 第3步：创建文件夹结构 (Create Directories)
# python
# # 提示：双循环。外层循环target_dirs的每个拆分，内层循环source_dirs的每个类别
# # 使用 os.makedirs(路径, exist_ok=True)
for class_name,source_dir in source_dir.items():
    all_images = [f for f in os.listdir(source_dir)
    if f.lower().endswith((".png",".jpg",".bmp"))]

    train_file,temp_file = train_test_split(
        all_images,
        train_size=train_traio,
        random_state=42,
        shuffle=True
    )

    val_file,test_file = train_test_split(
        temp_file,
        train_size=val_traio/(val_traio+test_traio),
        random_state=42,
        shuffle=True
    )

    for f in train_file:
        src = os.path.join(source_dir,f)
        dst = os.path.join(taget_dir["train"],class_name)
        shutil.copy2(src, dst)

    for f in val_file:
        src = os.path.join(source_dir,f)
        dst = os.path.join(taget_dir["val"],class_name)
        shutil.copy2(src, dst)

    for f in test_file:
        src = os.path.join(source_dir,f)
        dst = os.path.join(taget_dir["test"],class_name)
        shutil.copy2(src, dst)

print("文件分类完成，开始数据划分")

# 复制文件到对应文件
# # 提示：大循环 for class_name, source_dir in source_dirs.items():
#     # 4.1 用os.listdir和条件判断列出所有图片 -> all_images
#     # 4.2 第一次train_test_split: 分出train_files和temp_files (train_size=train_ratio)
#     # 4.3 第二次train_test_split: 从temp_files中分出val_files和test_files (计算比例)
#     # 4.4 三个小循环：分别将train_files, val_files, test_files复制到对应目录
#     #     在循环内：用os.path.join拼src和dst，用shutil.copy2复制
# 第5步：设置参数 & 数据生成器 (Parameters & Data Generators)
img_height,img_width = 225,225
input_shape = img_height,img_width,3
num_class = 5
batch_size = 16

# python
# # 提示：
# # 5.1 设置 img_height, img_width, batch_size, num_classes
# # 5.2 创建 train_datagen (带数据增强) 和 val_test_datagen (只归一化)
# # 5.3 用 .flow_from_directory 创建 train_generator, val_generator, test_generator
# #     注意参数：target_size, batch_size, class_mode='sparse'

train_test_datagen =tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    height_shift_range=0.5,
    width_shift_range=0.5,
    zoom_range=0.2
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_test_datagen.flow_from_directory(
    taget_dir["train"],
    target_size = (img_height,img_width),
    batch_size=batch_size,
    class_mode="sparse"
)

val_generator = val_test_datagen.flow_from_directory(
    taget_dir["val"],
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode="sparse"
)

test_generator = val_test_datagen.flow_from_directory(
    taget_dir["test"],
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

print("数据划分完成")

# 第6步：构建模型 (Build Model)
# python
# # 提示：models.Sequential() 然后 .add() 各种层
# # 经典模式：Conv2D -> BatchNorm -> MaxPool -> Dropout -> ... -> Flatten/GAP -> Dense -> Output
# # 输出层：Dense(num_classes, activation='softmax')
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation = "relu",kernel_regularizer=regularizers.l2(0.005),input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64,(3,3),activation="relu",kernel_regularizer=regularizers.l2(0.005)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128,(3,3),activation="relu",kernel_regularizer=regularizers.l2(0.005)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.4))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_class,activation = "softmax"))

class_weights = {
    0:1.0,
    1:1.07,
    2:1.0,
    3:1.0,
    4:1.03
}

# 第7步：编译与训练 (Compile & Train)
# python
# 模型编译
model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# # 提示：
# # 7.1 model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 数据回调
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# # 7.2 定义 EarlyStopping 回调 (monitor='val_accuracy', restore_best_weights=True)
# # 7.3 model.fit(训练数据, validation_data=验证数据, callbacks=[回调], epochs=很多)
history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 20,
    callbacks = early_stop,
    class_weight = class_weights
)


# 第8步：评估与保存 (Evaluate & Save)
model.save(" shuiguo mode.h5")
print("保存至shuiguomode.h5")
# 1. 基础指标
test_generator.reset()
# 2. 精确率、召回率、F1（核心指标）
# 8.1 重置测试数据生成器，确保从开始评估
# 函数：test_generator.reset() - 重置生成器指针到起始位置
test_loss,test_acc = model.evaluate(test_generator)
print(f"准确率为{test_acc}|损失率为{test_loss}")


# 8.2 计算基础指标：测试准确率和损失率
# 函数：model.evaluate() - 返回[损失值, 准确率]
# 准确率(accuracy): 整体分类正确的比例 = 正确预测数/总预测数


# 8.3 获取真实标签和预测结果
# 函数：test_generator.classes - 获取真实标签
# 函数：model.predict().argmax(axis=1) - 预测并取概率最大的类别
y_ture = test_generator.classes
y_pred = model.predict(test_generator).argmax(axis = 1)
# 8.4 生成详细分类报告
# 函数：classification_report() - 生成精确率、召回率、F1分数报告
print(classification_report(y_ture,y_pred,target_names=CLASS_NAMES))

# 精确率(precision): 预测为某类中确实为该类的比例 = TP/(TP+FP)
# 召回率(recall): 实际某类中被正确预测的比例 = TP/(TP+FN)
# F1分数: 精确率和召回率的调和平均 = 2*(precision*recall)/(precision+recall)
test_generator.reset()
cm = confusion_matrix(y_ture,y_pred)
ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(cmap="Blues")
plt.title("混淆矩阵")
plt.show()
# 8.5 混淆矩阵可视化
# 函数：confusion_matrix() - 计算混淆矩阵
# 函数：ConfusionMatrixDisplay().plot() - 可视化混淆矩阵
# 混淆矩阵: 显示模型在各个类别间的混淆情况，定位具体分类问题
