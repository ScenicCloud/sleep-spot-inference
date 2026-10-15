"""
SleepSpot Plan B 实验：V0 engine + 手动模拟 spot 场景
=====================================================

场景定义：
1. Cold Start（冷启动，基准）: 
   关机 → 启动 vLLM → 模型加载 → 第一次可用推理的时间

2. Quick Restart（spot 被回收但立刻拉起新实例）:
   kill vLLM → 立刻重启 vLLM → 模型加载 → 第一次可用推理的时间
   （模拟 spot 回收通知到来后 30-60s 内被新实例接管）

3. Sleep-Mode Simulation（我们的方案，手动模拟）:
   vLLM 运行中 → kill（模型权重可能还在 RAM 热缓存中）→ 立刻重启 → 第一次可用推理
   （模拟 vLLM sleep mode：权重从 GPU 转到 CPU，然后快速恢复到 GPU）

4. TTL 过期重启（对照）:
   kill vLLM → sleep 30s（让 RAM 缓存冷却）→ 重启 → 第一次可用推理

每种场景跑 3 次取平均。
"""

import time, os, subprocess, sys, json, signal

MODEL = "/root/models/Qwen2.5-1.5B-Instruct"
VLLM_PORT_BASE = 8700
RESULTS_FILE = "/root/plan_b_results.json"
LOG_FILE = "/root/plan_b_experiment.log"
VLLM_CMD = "vllm serve %s --gpu-memory-utilization 0.6 --max-model-len 512 --max-num-seqs 16" % MODEL
STARTUP_WAIT = 45  # 等模型加载的秒数

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

def curl_health(port):
    cmd = "curl -s --max-time 5 http://localhost:%d/health" % port
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def curl_infer(port):
    import json as _json
    payload = _json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "say hi"}],
        "max_tokens": 10
    })
    cmd = ('curl -s --max-time 30 http://localhost:%d/v1/chat/completions '
           '-H "Content-Type: application/json" -d \'%s\'' % (port, payload))
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def wait_ready(port, max_wait=90):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = curl_health(port)
        if r.returncode == 0 and r.stdout.strip():
            return True, time.time() - t0
        time.sleep(3)
    return False, time.time() - t0

def kill_vllm(port):
    subprocess.run("pkill -9 -f 'vllm serve.*--port %d'" % port, shell=True, capture_output=True)
    # 也 kill 所有 vllm serve
    subprocess.run("pkill -9 -f 'vllm serve'", shell=True, capture_output=True)
    time.sleep(3)

def run_trial(port, scenario_name, scenario_action_fn, trial_num):
    """跑单次实验，返回 (启动时间s, 推理TTFTs)"""
    log("\n--- %s 第 %d 次 ---" % (scenario_name, trial_num))
    
    # 执行场景前置动作
    scenario_action_fn()
    
    # 启动 vLLM
    t_start = time.time()
    proc = subprocess.Popen(VLLM_CMD % port + " > /tmp/vllm_%d.log 2>&1" % port, shell=True)
    
    # 等 ready
    ready, startup_time = wait_ready(port)
    
    if not ready:
        log("  FAILED: 没 ready")
        proc.kill()
        return None
    
    t_ready = time.time()
    
    # 预热推理
    r = curl_infer(port)
    if r.returncode != 0:
        log("  FAILED: 推理失败")
        kill_vllm(port)
        return None
    
    # 测 TTFT
    t_ttft_start = time.time()
    r2 = curl_infer(port)
    ttft = time.time() - t_ttft_start
    
    log("  启动时间: %.2fs  TTFT: %.2fs  总时间: %.2fs" % (startup_time, ttft, startup_time + ttft))
    
    kill_vllm(port)
    
    return {
        "startup_time": round(startup_time, 2),
        "ttft": round(ttft, 2),
        "total": round(startup_time + ttft, 2)
    }

def scenario_cold_start():
    """彻底冷启动：先 kill + 等 RAM 释放"""
    kill_vllm(9999)  # kill 所有
    time.sleep(5)

def scenario_quick_restart():
    """模拟 spot 回收通知 + 立刻拉新"""
    kill_vllm(9999)
    time.sleep(1)  # 极短间隔

def scenario_sleep_sim():
    """模拟 sleep mode：快速重启，利用 CPU 缓存"""
    kill_vllm(9999)
    time.sleep(0.5)  # 最短间隔，模拟 wake from sleep

def scenario_ttl_restart():
    """TTL 过期重启：等 RAM 缓存冷却"""
    kill_vllm(9999)
    time.sleep(30)

def main():
    log("=" * 60)
    log("SleepSpot Plan B 实验")
    log("时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log("模型: " + MODEL)
    log("=" * 60)
    
    # 确保干净开始
    kill_vllm(9999)
    time.sleep(3)
    
    results = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": {}
    }
    
    scenarios = [
        ("Cold Start (基准)", scenario_cold_start, 0),
        ("Sleep-Mode Sim (我们的方案)", scenario_sleep_sim, 1),
        ("Quick Restart (spot 紧急拉新)", scenario_quick_restart, 2),
        ("TTL Restart (对照)", scenario_ttl_restart, 3),
    ]
    
    for name, action, port_offset in scenarios:
        port = VLLM_PORT_BASE + port_offset
        results["trials"][name] = []
        for t in range(3):
            r = run_trial(port, name, action, t + 1)
            if r:
                results["trials"][name].append(r)
    
    # 汇总
    log("\n" + "=" * 60)
    log("实验结果汇总")
    log("=" * 60)
    
    cold_total = None
    for name, trials in results["trials"].items():
        if trials:
            avg_start = sum(t["startup_time"] for t in trials) / len(trials)
            avg_ttft = sum(t["ttft"] for t in trials) / len(trials)
            avg_total = sum(t["total"] for t in trials) / len(trials)
            results["trials"][name + "_avg"] = {
                "startup_time": round(avg_start, 2),
                "ttft": round(avg_ttft, 2),
                "total": round(avg_total, 2)
            }
            log("%-35s 启动: %.1fs  TTFT: %.1fs  总: %.1fs" % (name, avg_start, avg_ttft, avg_total))
            
            if "Cold Start" in name:
                cold_total = avg_total
    
    log("")
    log("加速比对比 (相对冷启动):")
    if cold_total:
        for name, trials in results["trials"].items():
            if "_avg" not in name and trials and "Cold Start" not in name:
                avg = sum(t["total"] for t in trials) / len(trials)
                speedup = cold_total / avg
                log("  %-30s %.2fx" % (name, speedup))
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log("\nJSON 结果: " + RESULTS_FILE)
    log("=" * 60)

if __name__ == "__main__":
    main()
