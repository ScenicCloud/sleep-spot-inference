"""
SleepSpot: vLLM Sleep Mode + Spot Preemption 模拟实验
======================================================

在 Mac 上模拟 vLLM sleep mode 与 spot 回收场景。
不需要 NVIDIA GPU——用 sklearn 的文本分类模型模拟 LLM 加载/卸载。

核心模拟逻辑：
  - 冷启动（naive）：kill 进程 → 新进程从零加载模型
  - SleepSpot-L1 ：收到回收通知 → sleep（模型权重卸到 RAM）→ kill → 新进程从 RAM 恢复
  - SleepSpot-L2 ：收到回收通知 → sleep（模型权重落盘）→ kill → 新进程从磁盘恢复
"""

import os
import sys
import time
import pickle
import signal
import tempfile
import subprocess
import statistics
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 第一步：定义一个"假的 LLM"——模拟加载/卸载的时间成本
# ---------------------------------------------------------------------------

SIMULATED_MODEL_SIZE_MB = 500  # 模拟一个 500MB 的"模型权重"
LOAD_FROM_DISK_TIME = 4.0       # 冷启动：从磁盘加载 → 4s
LOAD_FROM_RAM_TIME = 0.3        # 从 RAM 恢复 → 0.3s（快 13x）
SLEEP_TO_RAM_TIME = 0.5         # sleep L1：GPU→RAM → 0.5s
SLEEP_TO_DISK_TIME = 1.0        # sleep L2：完全落盘 → 1.0s
GPU_INIT_TIME = 2.0             # 模拟 GPU 初始化（vLLM 启动开销）


class FakeLLM:
    """模拟一个 LLM 推理引擎。"""

    def __init__(self, model_size_mb=SIMULATED_MODEL_SIZE_MB):
        self.model_size = model_size_mb
        self.model_loaded = False
        self.weights = None  # 模拟模型权重

    def load_from_disk(self):
        """模拟冷启动：从零加载模型。"""
        print(f"  [FakeLLM] 冷启动：GPU 初始化 + 从磁盘加载 {self.model_size}MB 权重...")
        time.sleep(GPU_INIT_TIME)
        time.sleep(LOAD_FROM_DISK_TIME)
        self.weights = "x" * self.model_size * 1024 * 1024  # 分配实际内存
        self.model_loaded = True
        print(f"  [FakeLLM] 加载完成，总耗时 {GPU_INIT_TIME + LOAD_FROM_DISK_TIME:.2f}s")

    def sleep_to_ram(self):
        """Sleep Level 1：权重从 GPU 卸到 CPU RAM。"""
        print(f"  [FakeLLM] Sleep L1：权重 {self.model_size}MB 从 GPU → RAM...")
        time.sleep(SLEEP_TO_RAM_TIME)
        # 权重还在 self.weights 里（RAM），GPU 释放了
        self.model_loaded = False  # GPU 上没有了
        print(f"  [FakeLLM] Sleep L1 完成，耗时 {SLEEP_TO_RAM_TIME:.2f}s")

    def sleep_to_disk(self, path):
        """Sleep Level 2：权重完全释放并落盘。"""
        print(f"  [FakeLLM] Sleep L2：权重 {self.model_size}MB 完全释放并写入磁盘...")
        time.sleep(SLEEP_TO_DISK_TIME)
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)
        self.weights = None
        self.model_loaded = False
        print(f"  [FakeLLM] Sleep L2 完成，耗时 {SLEEP_TO_DISK_TIME:.2f}s")

    def wake_from_ram(self):
        """从 Sleep L1 状态唤醒：RAM → GPU。"""
        print(f"  [FakeLLM] Wake L1：权重 {self.model_size}MB 从 RAM → GPU...")
        time.sleep(GPU_INIT_TIME * 0.5)  # 不需要完整 GPU init
        time.sleep(LOAD_FROM_RAM_TIME)
        self.model_loaded = True
        print(f"  [FakeLLM] Wake L1 完成，耗时 {GPU_INIT_TIME * 0.5 + LOAD_FROM_RAM_TIME:.2f}s")

    def wake_from_disk(self, path):
        """从 Sleep L2 状态唤醒：磁盘 → RAM → GPU。"""
        print(f"  [FakeLLM] Wake L2：权重从磁盘加载 → RAM → GPU...")
        time.sleep(GPU_INIT_TIME * 0.5)
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
        time.sleep(LOAD_FROM_RAM_TIME * 1.2)  # 比纯 RAM 稍慢
        self.model_loaded = True
        print(f"  [FakeLLM] Wake L2 完成，耗时 {GPU_INIT_TIME * 0.5 + LOAD_FROM_RAM_TIME * 1.2:.2f}s")

    def infer(self, prompt="hello"):
        """模拟一次推理。"""
        if not self.model_loaded:
            raise RuntimeError("模型未加载，无法推理")
        time.sleep(0.1)  # 模拟推理耗时
        return f"response for '{prompt}'"


# ---------------------------------------------------------------------------
# 第二步：三种场景的恢复流程
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    recovery_time: float
    breakdown: dict


def scenario_naive():
    """
    Baseline A: Naive 方案
    流程：spot 被回收 → 新 spot 冷启动 → 服务恢复
    """
    print("\n" + "=" * 60)
    print("场景 A: Naive（什么都不做）")
    print("=" * 60)

    # 模拟旧实例正在运行
    llm = FakeLLM()
    llm.load_from_disk()

    # 旧实例被 spot 回收
    print("\n  [Spot] 实例被回收，进程终止...")
    time.sleep(0.1)

    # 新 spot 实例拉起，冷启动
    print("\n  [Spot] 新实例拉起，开始冷启动...")
    t0 = time.time()
    new_llm = FakeLLM()
    new_llm.load_from_disk()
    recovery_time = time.time() - t0

    print(f"\n  ✅ 恢复完成。总恢复时间: {recovery_time:.2f}s")

    return ExperimentResult(
        name="Naive (冷启动)",
        recovery_time=recovery_time,
        breakdown={
            "step": ["新实例拉起", "GPU 初始化", "模型加载"],
            "time_s": [0.1, GPU_INIT_TIME, LOAD_FROM_DISK_TIME]
        }
    )


def scenario_sleep_spot_l1():
    """
    SleepSpot-L1 方案
    流程：收到回收通知 → sleep L1 → 旧实例被回收 → 新实例从 RAM 恢复
    """
    print("\n" + "=" * 60)
    print("场景 B: SleepSpot-L1（RAM 保持权重）")
    print("=" * 60)

    # 模拟旧实例正在运行
    llm = FakeLLM()
    llm.load_from_disk()

    # 收到回收通知 → 触发 sleep L1
    print("\n  [Spot] 收到回收通知（grace period 30s）→ 触发 sleep L1")
    llm.sleep_to_ram()

    # 旧实例被 spot 回收
    print("\n  [Spot] 实例被回收，进程终止...")
    time.sleep(0.1)

    # 新 spot 实例拉起 → 从 sleep 状态唤醒
    print("\n  [Spot] 新实例拉起，从 RAM 唤醒模型...")
    t0 = time.time()
    new_llm = FakeLLM()
    new_llm.weights = "x" * SIMULATED_MODEL_SIZE_MB * 1024 * 1024  # 模拟：状态已经在 RAM 里
    new_llm.wake_from_ram()
    recovery_time = time.time() - t0

    print(f"\n  ✅ 恢复完成。总恢复时间: {recovery_time:.2f}s")

    return ExperimentResult(
        name="SleepSpot-L1 (RAM 恢复)",
        recovery_time=recovery_time,
        breakdown={
            "step": ["新实例拉起", "GPU 初始化(减半)", "RAM→GPU 加载"],
            "time_s": [0.1, GPU_INIT_TIME * 0.5, LOAD_FROM_RAM_TIME]
        }
    )


def scenario_sleep_spot_l2():
    """
    SleepSpot-L2 方案
    流程：收到回收通知 → sleep L2（落盘）→ 旧实例被回收 → 新实例从磁盘恢复
    """
    print("\n" + "=" * 60)
    print("场景 C: SleepSpot-L2（磁盘保持权重）")
    print("=" * 60)

    # 模拟旧实例正在运行
    llm = FakeLLM()
    llm.load_from_disk()

    # 创建临时文件作为 sleep 的存储位置
    tmp_dir = tempfile.mkdtemp()
    sleep_file = Path(tmp_dir) / "model_sleep.pkl"

    # 收到回收通知 → 触发 sleep L2
    print("\n  [Spot] 收到回收通知（grace period 30s）→ 触发 sleep L2")
    llm.sleep_to_disk(str(sleep_file))

    # 旧实例被 spot 回收
    print("\n  [Spot] 实例被回收，进程终止...")
    time.sleep(0.1)

    # 新 spot 实例拉起 → 从磁盘唤醒
    print("\n  [Spot] 新实例拉起，从磁盘唤醒模型...")
    t0 = time.time()
    new_llm = FakeLLM()
    new_llm.wake_from_disk(str(sleep_file))
    recovery_time = time.time() - t0

    print(f"\n  ✅ 恢复完成。总恢复时间: {recovery_time:.2f}s")

    # 清理
    sleep_file.unlink(missing_ok=True)
    Path(tmp_dir).rmdir()

    return ExperimentResult(
        name="SleepSpot-L2 (磁盘恢复)",
        recovery_time=recovery_time,
        breakdown={
            "step": ["新实例拉起", "GPU 初始化(减半)", "磁盘→RAM→GPU"],
            "time_s": [0.1, GPU_INIT_TIME * 0.5, LOAD_FROM_RAM_TIME * 1.2]
        }
    )


# ---------------------------------------------------------------------------
# 第三步：主实验入口
# ---------------------------------------------------------------------------

def run_all_experiments(n_trials=3):
    """多次运行取平均值，减少波动。"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SleepSpot: Sleep Mode + Spot Preemption 模拟实验            ║")
    print("║  模拟参数：模型 500MB，n_trials =", n_trials, "                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = {
        "Naive": [],
        "SleepSpot-L1": [],
        "SleepSpot-L2": [],
    }

    for trial in range(n_trials):
        print(f"\n\n{'█' * 10} Trial {trial + 1}/{n_trials} {'█' * 10}")

        r = scenario_naive()
        results["Naive"].append(r.recovery_time)

        r = scenario_sleep_spot_l1()
        results["SleepSpot-L1"].append(r.recovery_time)

        r = scenario_sleep_spot_l2()
        results["SleepSpot-L2"].append(r.recovery_time)

    # 汇总输出
    print("\n\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                     实验结果汇总                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    naive_mean = statistics.mean(results["Naive"])
    l1_mean = statistics.mean(results["SleepSpot-L1"])
    l2_mean = statistics.mean(results["SleepSpot-L2"])

    print(f"\n  {'方案':<25} {'平均恢复时间':<15} {'比 Naive 快':<15}")
    print(f"  {'-' * 55}")
    print(f"  {'Naive (冷启动)':<25} {naive_mean:<15.2f}s {'1.00x (基准)':<15}")
    print(f"  {'SleepSpot-L1':<25} {l1_mean:<15.2f}s {naive_mean / l1_mean:<15.1f}x")
    print(f"  {'SleepSpot-L2':<25} {l2_mean:<15.2f}s {naive_mean / l2_mean:<15.1f}x")

    print(f"\n  🎯 SleepSpot-L1 相比 Naive 快了 {naive_mean - l1_mean:.2f}s "
          f"（约 {(naive_mean / l1_mean):.1f} 倍）")

    # 保存结果
    results_path = Path(__file__).parent.parent / "results" / "simulation_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"Naive: {naive_mean:.3f}s\n")
        f.write(f"SleepSpot-L1: {l1_mean:.3f}s\n")
        f.write(f"SleepSpot-L2: {l2_mean:.3f}s\n")
        f.write(f"L1 speedup: {naive_mean / l1_mean:.1f}x\n")
        f.write(f"L2 speedup: {naive_mean / l2_mean:.1f}x\n")
    print(f"\n  📁 结果已保存到: {results_path}")

    return results


if __name__ == "__main__":
    run_all_experiments(n_trials=3)
