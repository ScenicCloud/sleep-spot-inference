# 云服务器 GPU 环境搭建指南

## 1. 云平台选择

| 平台 | 推荐 GPU | 学生价 | 说明 |
|------|---------|--------|------|
| **AutoDL** | RTX 4090 24GB | ~1-2元/小时 | 国内学生最友好，支付宝付款 |
| **矩池云** | RTX 4090 24GB | ~1.5-2.5元/小时 | 类似 AutoDL，界面简洁 |
| 阿里云 ECS | L4 / A10 | 较贵 | 需要实名认证 |
| AWS | g5.xlarge | ~$0.786/hr | 海外，需信用卡 |

**推荐 AutoDL**（http://www.autodl.com），注册 → 实名认证 → 租一台 4090 → SSH 连接。

> 预估总花费：实验时间 ~10-15 小时，花费 15-30 元。

## 2. 连接云服务器

```bash
# 本地 Mac 上用 SSH 连接
ssh root@<云服务器IP> -p <端口号>

# 或者用 AutoDL 提供的 JupyterLab / VSCode Web
```

## 3. 环境搭建（复制粘贴执行）

### 3.1 基础环境

```bash
# 更新系统
apt update && apt upgrade -y

# 检查 CUDA（AutoDL 通常预装了 CUDA 12.x）
nvidia-smi
```

期望看到类似输出：
```
NVIDIA-SMI 535.xx   Driver Version: 535.xx   CUDA Version: 12.1
GPU  Name        Persistence-M  Bus Id     Disp.A  Volatile Uncorr. ECC
0    NVIDIA GeForce RTX 4090   Off   00000000:00:00.0  Off      0     0
```

### 3.2 创建 Python 环境

```bash
# 创建 conda 环境（AutoDL 默认有 conda）
conda create -n sleepspot python=3.11 -y
conda activate sleepspot

# 升级 pip
pip install --upgrade pip
```

### 3.3 安装 PyTorch + CUDA

```bash
# 检查 CUDA 版本
nvcc --version

# 根据 CUDA 版本安装 PyTorch
# CUDA 12.1 用下面这行
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 用这个
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

期望输出：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### 3.4 安装 vLLM（关键！）

```bash
# 安装 vLLM（pip 自动匹配 CUDA 版本）
pip install vllm

# 验证安装
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

**关键检查**：vLLM 版本需要 >= 0.7.3 才有 sleep mode。如果太旧：
```bash
pip install --upgrade vllm
```

### 3.5 安装 HuggingFace 依赖

```bash
pip install transformers accelerate sentencepiece
pip install huggingface_hub

# （可选）登录 HuggingFace 下载受限模型
# huggingface-cli login
```

### 3.6 下载小模型（先测通）

```bash
# 下载 Qwen2.5-1.5B（最小的，几分钟下完）
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
print(f'Downloading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Done!')
"
```

或者用 huggingface-cli：
```bash
# 只下载（不加载到内存）
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
```

## 4. 验证 Sleep Mode 功能

### 4.1 启动 vLLM 服务（带 sleep mode）

```bash
# 启动 vLLM，开启 sleep mode
# 注意：只支持 CUDA 平台，用 --enable-sleep-mode 参数
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 &

# 等几秒让它启动
sleep 10

# 测试基本推理
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "你好，介绍一下自己"}],
    "max_tokens": 50
  }'
```

### 4.2 测试 Sleep API

```bash
# 触发 sleep（Level 1：权重保留在 RAM）
curl -X POST http://localhost:8000/v1/engine/sleep \
  -H "Content-Type: application/json" \
  -d '{"level": 1}'

# 检查 GPU 内存（应该几乎为 0）
nvidia-smi

# 唤醒
curl -X POST http://localhost:8000/v1/engine/wake \
  -H "Content-Type: application/json"

# 再检查 GPU 内存（应该恢复）
nvidia-smi

# 测试推理是否恢复
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "sleep 恢复后还能正常工作吗？"}],
    "max_tokens": 50
  }'
```

如果这个能跑通，环境就 OK 了。

## 5. 停止服务 + 释放资源

```bash
# 找到 vLLM 进程并 kill
pkill -f vllm

# 确认 GPU 已释放
nvidia-smi
```

## 6. 常见问题

### Q: nvidia-smi 显示 driver 版本太低？
```bash
# AutoDL/矩池云 不会有这个问题，他们都预装了
# 自己装的话用：
# apt install nvidia-driver-535
```

### Q: pip install vllm 报错？
```bash
# 确保 PyTorch 是 CUDA 版本，不是 CPU 版本
python -c "import torch; print(torch.version.cuda)"
# 如果是 None，说明装了 CPU 版，重装：
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q: vLLM 启动报 sleep mode 不支持？
```bash
# 检查 vLLM 版本
pip show vllm
# 需要 >= 0.7.3
pip install --upgrade vllm
```

### Q: OOM (Out of Memory)？
```bash
# 减小 GPU 内存使用比例
vllm serve <model> --gpu-memory-utilization 0.6
# 或者用更小的模型
```

### Q: 模型下载太慢？
```bash
# 设置 HuggingFace 镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com
```
