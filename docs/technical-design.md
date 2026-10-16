# SleepSpot: Sleep Mode + Spot Instance Elastic LLM Inference

## 1. 问题陈述

### 1.1 Serverless LLM 推理的现状

LLM 推理服务正在向 serverless 架构迁移，因为它提供了按需付费、自动扩缩容的经济优势。然而：

- 冷启动问题（cold start latency）：函数闲置后首次请求需要从零加载模型权重（几十 GB），耗时几十秒到上百秒
- Spot 实例回收问题：使用低成本 spot 实例时，云厂商随时可能回收资源，导致服务中断

### 1.2 本文要解决的具体问题

> **当 spot 实例被回收通知时，如何最小化推理服务的中断时间？**

场景：

1. 用户用 spot 实例跑 serverless LLM 推理
2. 云厂商发出回收通知（30-60 秒 grace period）
3. 旧实例终止，新 spot 实例需要立刻接手
4. 从旧实例终止到新实例恢复服务的时间 = 我们要优化的指标

## 2. 现有方案及其不足

### 2.1 现有方案 1：SpotServe (ASPLOS'24)

SpotServe 是第一个解决 spot 实例上 LLM 推理问题的工作。

**核心机制**：stateful inference recovery（有状态推理恢复）
- 在每个 token 生成后做 checkpoint，把 KV cache 迁移到别的实例
- spot 被回收后，新实例从其他实例拉回 KV cache 继续推理

**不足**：
1. 实现复杂——需要深度修改 vLLM 引擎，加入 token 粒度的 commit/checkpoint
2. 依赖有其他存活的实例来承接 KV cache
3. 没有利用 vLLM 2025 年才出现的 sleep mode 功能

### 2.2 现有方案 2：SkyServe (2024)

**核心机制**：跨区域 spot+on-demand 混合调度

**不足**：聚焦在实例 provisioning 层面，没有解决单个 spot 实例被回收后的快速恢复问题

### 2.3 现有方案 3：什么都不做（naive）

spot 被回收 → 新 spot 冷启动加载模型 → 几十秒的服务中断

### 2.4 现有方案 4：仅用 on-demand 实例

贵很多，失去了 spot 的成本优势

## 3. 我们的方案：SleepSpot

### 3.1 核心洞察

**vLLM 的 sleep mode（2025 年 3 月 v0.7.3 引入，2025 年 5 月 V1 引擎支持）提供了一种"快速暂停/快速唤醒"模型的能力，但没有任何论文把它和 spot 实例回收场景结合起来。**

- Sleep mode Level 1：模型权重从 GPU 卸到 CPU RAM（唤醒时从 RAM 加载回 GPU，几秒）
- Sleep mode Level 2：模型权重完全释放（唤醒时从磁盘加载，十几秒）
- 官方博客声称比全量加载快 18-200x [cite:vllm-blog]

### 3.2 方案设计

```
正常运行中
  │
  ▼
收到 spot 回收通知（grace period 开始）
  │
  ▼
【SleepSpot 协调器】触发 vLLM sleep mode
  ├── Level 1（推荐）: 权重 → CPU RAM
  └── Level 2: 权重 → 完全释放
  │
  ▼
Spot 实例被回收（进程终止）
  │
  ▼
新 spot 实例被拉起
  │
  ▼
【SleepSpot 协调器】启动 vLLM 并触发 wake up
  │
  ▼
服务恢复（从 sleep 状态唤醒，不是从零加载）
```

### 3.3 与 SpotServe 的对比

| 维度 | SpotServe | SleepSpot（我们） |
|------|-----------|-------------------|
| 恢复机制 | token 粒度 KV cache checkpoint + 跨实例迁移 | sleep mode 状态保持 + 快速唤醒 |
| 实现复杂度 | 高（修改 vLLM 引擎核心） | 低（只调用 vLLM sleep/wake API） |
| 恢复依赖 | 需要其他存活实例 | 只需 CPU RAM（Level 1）或磁盘（Level 2） |
| 核心改动点 | 推理引擎内部 | 推理引擎外部的协调器 |
| 解决的问题 | 推理过程中 spot 被回收的 KV cache 恢复 | spot 被回收后新实例的快速启动 |
| 两者关系 | 互补而非替代 | 互补而非替代 |

### 3.4 我们的贡献声明

> **We propose SleepSpot, a lightweight coordination layer that leverages vLLM's sleep mode to accelerate service recovery after spot instance preemption. Unlike SpotServe which requires deep engine modifications and cross-instance KV cache migration, SleepSpot uses only the public vLLM sleep/wake APIs and achieves comparable recovery speed with orders of magnitude less implementation complexity.**

## 4. 实验计划

### 4.1 对比基线

- **Baseline A (Naive)**：spot 被回收后，新实例从零冷启动加载模型
- **Baseline B (SpotServe-style)**：SpotServe 的 stateful recovery 机制（对比用）
- **SleepSpot-L1 (我们)**：sleep mode Level 1（CPU RAM 保持权重）
- **SleepSpot-L2 (我们)**：sleep mode Level 2（完全释放权重）

### 4.2 测试场景

- 场景 1：spot 被回收，新 spot 同机器（模拟——同一进程内的 kill + restart）
- 场景 2：spot 被回收，新 spot 不同机器（跨机器的状态迁移）
- 场景 3：spot 被频繁回收（高 churn 场景）

### 4.3 指标

- **Recovery Time**：spot 回收 → 新实例开始服务的总时间
- **TTFT**：新实例上第一个 token 的时间
- **P99 Latency**：恢复后推理请求的尾延迟
- **Cost**：spot+sleep vs spot+naive 的成本差异
- **Accuracy**：恢复后推理结果的准确性（确保 sleep/wake 不影响正确性）

### 4.4 实验平台

- Phase 1（云 GPU）：真实 vLLM + CUDA 上的 sleep mode 实验
  - 推荐云平台：AutoDL、矩池云、阿里云 ECS（有学生折扣）
  - 推荐 GPU：RTX 4090 24GB 或 L4 24GB（单卡足够）

### 4.5 模型选择

| 模型 | 大小 | 显存占用 | 说明 |
|------|------|---------|------|
| Qwen2.5-1.5B | 1.5B | ~3GB | 最小，冷启动快 |
| Llama-3.2-3B | 3B | ~6GB | 中等 |
| Qwen2.5-7B | 7B | ~14GB | 主流，需 24GB 卡 |

## 5. 论文结构（预写）

```
Title: SleepSpot: Accelerating Serverless LLM Service Recovery after Spot Preemption

Abstract
1. Introduction
   1.1 Motivation (spot instances + cold start problem)
   1.2 Key insight (vLLM sleep mode + spot preemption = no one has done this)
   1.3 Contributions

2. Background
   2.1 Spot instances and preemption
   2.2 Serverless LLM inference
   2.3 vLLM sleep mode (introduced 2025, brief explanation)

3. Related Work
   3.1 SpotServe (ASPLOS'24)
   3.2 SkyServe
   3.3 ServerlessLLM

4. Design
   4.1 Overview
   4.2 SleepSpot Coordinator
   4.3 Sleep Level Selection Strategy
   4.4 Failure Modes and Handling

5. Implementation

6. Evaluation
   6.1 Setup
   6.2 Main Results (recovery time comparison)
   6.3 Micro-benchmarks
   6.4 Cost Analysis
   6.5 Discussion

7. Limitations

8. Conclusion

References
```

## 6. 时间表

| 周 | 任务 |
|----|------|
| W1 | 方案调研与实验设计 |
| W2 | 云 GPU 环境 + vLLM sleep mode 实验 |
| W3 | 完整对比实验 + 数据整理 |
| W4 | 写论文 + 润色 |

## 7. 参考文献

- [cite:vllm-blog] vLLM Sleep Mode blog (2025-10-26): https://blog.vllm.ai/2025/10/26/sleep-mode.html
- [cite:SpotServe] SpotServe (ASPLOS'24): https://arxiv.org/html/2311.15566v1
- [cite:SkyServe] SkyServe: https://arxiv.org/pdf/2411.01438
- [cite:ServerlessLLM] ServerlessLLM (OSDI'24): https://arxiv.org/pdf/2401.14351v2
