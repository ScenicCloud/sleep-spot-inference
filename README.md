# SleepSpot

**基于 vLLM Sleep Mode 的 Spot 实例大模型推理快速恢复方案**

本仓库是论文 *SleepSpot: Fast Recovery for LLM Inference on Preemptible Instances via vLLM Sleep Mode* 的配套代码与数据。核心发现：

- **恢复时间**：1.5B 模型冷启动 49.73s → Sleep L1 唤醒 **0.25s**（加速 ~200×）；7B 冷启动 54.09s → 唤醒 **0.85s**（加速 ~64–93×）
- **显存释放**：~95%（7B 从 20 GiB 降至 1.03 GiB）
- **版本局限发现**：vLLM 0.8.5 V1 引擎的 Sleep Level 2 未真正丢弃主机侧权重副本（双规模复现）

## 仓库结构

```
sleep-spot-inference/
├── paper/              # 论文初稿（Markdown）
├── docs/               # 技术方案设计文档
├── src/                # 实验脚本（可复现）
│   ├── sleep_mode_experiment.py      # 1.5B 主实验
│   ├── sleep_mode_experiment_7b.py   # 7B 主实验
│   ├── simulate_spot_recovery.py     # Mac 原型模拟
│   └── ...
├── results/            # 原始实验数据与日志
└── cloud-setup/        # 云 GPU 环境搭建指南
```

## 实验环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4090 24GB（单卡，矩池云） |
| 引擎 | vLLM 0.8.5（V1 Engine，`--enable-sleep-mode`） |
| 模型 | Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct（BF16） |

## 复现

```bash
# 1. 环境（详见 cloud-setup/setup-guide.md）
pip install vllm==0.8.5
export VLLM_SERVER_DEV_MODE=1   # 关键！开启 /sleep /wake_up 路由

# 2. 运行实验
python src/sleep_mode_experiment_7b.py
```

详细工程经验（路由门控、子进程清理、JIT 依赖等坑）见论文 5.5 节。

## 论文

[paper/paper_draft.md](paper/paper_draft.md)

## 许可

MIT（代码）；论文内容版权归作者。
