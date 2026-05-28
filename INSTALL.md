# Python依赖安装

## 方式一：使用国内镜像源（推荐）

### 清华源 (Tsinghua)
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 阿里源 (Aliyun)
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### 中科大源 (USTC)
```bash
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn
```

### 永久配置（一劳永逸）
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

## PyTorch 特别说明
PyTorch 安装较大（~2GB），建议使用清华/阿里镜像：
```bash
# CPU版本
pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/

# CUDA 11.8版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -i https://mirrors.aliyun.com/pypi/simple/
```

## 依赖清单

| 包名 | 最低版本 | 用途 |
|------|----------|------|
| torch | 2.0.0 | 深度学习框架 |
| torchvision | 0.15.0 | 计算机视觉工具库 |
| numpy | 1.24.0 | 数值计算 |
| opencv-python | 4.8.0 | 图像处理 |
| matplotlib | 3.7.0 | 绘图可视化 |
| seaborn | 0.12.0 | 统计可视化(热力图) |
| scikit-learn | 1.3.0 | 机器学习评估指标 |
| tqdm | 4.65.0 | 进度条 |
| pyyaml | 6.0 | 配置文件解析 |
| pillow | 10.0.0 | 图像读取 |
| tensorboard | 2.13.0 | 训练过程监控 |
