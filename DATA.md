# PlantVillage 数据集说明

## 概述

**PlantVillage** 是植物病害识别领域最广泛使用的公开基准数据集，由宾夕法尼亚州立大学(PSU)发布。

| 属性 | 数值 |
|------|------|
| 图像总数 | 54,305 张(原始) / 54,303 张(TFDS版) |
| 类别数 | 38 类 |
| 作物种类 | 14 种 |
| 图像格式 | JPG (RGB 彩色) |
| 图像尺寸 | 256×256 (原始) → 缩放至 224×224 |
| 拍摄环境 | 实验室统一背景 |
| 数据集大小 | ~827 MB (压缩) |
| 许可 | 开放获取 (CC0/Public Domain) |

## 包含的作物及病害类别

| 作物 | 病害类别 | 数量 |
|------|----------|------|
| Apple (苹果) | Scab, Black Rot, Cedar Rust, Healthy | 4 |
| Blueberry (蓝莓) | Healthy | 1 |
| Cherry (樱桃) | Powdery Mildew, Healthy | 2 |
| Corn (玉米) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| Grape (葡萄) | Black Rot, Esca(Black Measles), Leaf Blight, Healthy | 4 |
| Orange (橙子) | Huanglongbing(Citrus Greening) | 1 |
| Peach (桃子) | Bacterial Spot, Healthy | 2 |
| Pepper (辣椒) | Bacterial Spot, Healthy | 2 |
| Potato (土豆) | Early Blight, Late Blight, Healthy | 3 |
| Raspberry (覆盆子) | Healthy | 1 |
| Soybean (大豆) | Healthy | 1 |
| Squash (南瓜) | Powdery Mildew | 1 |
| Strawberry (草莓) | Leaf Scorch, Healthy | 2 |
| Tomato (番茄) | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |

## 数据集目录结构

```
data/PlantVillage/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry_(including_sour)___Powdery_mildew/
├── Cherry_(including_sour)___healthy/
├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
├── Corn_(maize)___Common_rust/
├── Corn_(maize)___Northern_Leaf_Blight/
├── Corn_(maize)___healthy/
├── Grape___Black_rot/
├── Grape___Esca_(Black_Measles)/
├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
├── Grape___healthy/
├── Orange___Haunglongbing_(Citrus_greening)/
├── Peach___Bacterial_spot/
├── Peach___healthy/
├── Pepper,_bell___Bacterial_spot/
├── Pepper,_bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Raspberry___healthy/
├── Soybean___healthy/
├── Squash___Powdery_mildew/
├── Strawberry___Leaf_scorch/
├── Strawberry___healthy/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites Two-spotted_spider_mite/
├── Tomato___Target_Spot/
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
├── Tomato___Tomato_mosaic_virus/
└── Tomato___healthy/
```

## 下载方式

### 方式一：GitHub 官方源（需科学上网）
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git --depth 1
mv PlantVillage-Dataset/raw/color/* data/PlantVillage/
rm -rf PlantVillage-Dataset
```

### 方式二：阿里云天池（国内直连，推荐）
访问 [阿里云天池 PlantVillage](https://tianchi.aliyun.com/dataset/160100)，直接下载。

### 方式三：Kaggle（需注册，速度较慢）
```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/PlantVillage/
```

### 方式四：TensorFlow Datasets（代码自动下载）
```python
import tensorflow_datasets as tfds
ds = tfds.load('plant_village', split='train', data_dir='./data/')
```

### 方式五：Hugging Face（国内可访问）
```bash
# 需安装 huggingface_hub
pip install huggingface_hub
# 下载数据集（需自行编写代码转换格式）
```

## 数据集预处理（本项目自动完成）

| 步骤 | 操作 | 参数 |
|------|------|------|
| 数据划分 | 训练:验证:测试 | 7:1.5:1.5 |
| 图像缩放 | 统一大小 | 224×224 |
| 归一化 | ImageNet 标准 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| 训练增强 | 随机水平翻转 | p=0.5 |
| 训练增强 | 随机旋转 | ±15° |
| 训练增强 | 色彩抖动 | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 |

## 参考文献

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.

## 数据集官方网站

- https://plantvillage.psu.edu/
- https://github.com/spMohanty/PlantVillage-Dataset
