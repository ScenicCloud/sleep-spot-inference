"""
SleepSpot 真实实验：vLLM Sleep Mode vs 冷启动 时间对比
=====================================================
"""

import time, os, subprocess, sys, json

MODEL_PATH = "/root/models/Qwen2.5-1.5B-Instruct"
RESULTS_FILE = "/root/sleepspot_results.json"
LOG_FILE = "/root/sleepspot_experiment.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

def run_cmd(cmd, timeout=300):
    log("  -> $ " + cmd[:120])
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr

def build_curl(port, payload_dict, endpoint="/v1/chat/completions"):
    """构造 curl 命令，避免 f-string 大括号冲突"""
    import json as _json
    payload = _json.dumps(payload_dict)
    return (
        "curl -s --max-time 30 http://localhost:%d%s " % (port, endpoint) +
        "-H 'Content-Type: application/json' " +
        "-d '%s'" % payload
    )

def start_vllm(port, sleep_mode=True):
    """后台启动 vLLM，返回 PID 文件路径"""
    log("启动 vLLM on port %d ..." % port)
    extra = " --enable-sleep-mode" if sleep_mode else ""
    cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && conda activate myconda && "
        "vllm serve %s --port %d --gpu-memory-utilization 0.6 "
        "--max-model-len 512 --max-num-seqs 16 --host 0.0.0.0%s "
        "> /root/vllm_%d.log 2>&1 & echo $!" % (MODEL_PATH, port, extra, port)
    )
    rc, out, err = run_cmd(cmd)
    pid = out.strip().split("\n")[-1]
    log("  PID: %s" % pid)
    time.sleep(28)  # 等模型加载
    return pid

def kill_pid(pid):
    run_cmd("kill %s 2>/dev/null; sleep 2" % pid)

def wait_ready(port, max_wait=60):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        rc, out, err = run_cmd("curl -s http://localhost:%d/health 2>/dev/null" % port, timeout=10)
        if rc == 0 and out.strip():
            return True
        time.sleep(2)
    return False

def measure_cold_start():
    log("\n===== 冷启动实验 =====")
    port = 8750
    t0 = time.time()
    pid = start_vllm(port, sleep_mode=False)
    
    # 等 ready
    wait_ready(port)
    
    elapsed = time.time() - t0
    log("冷启动时间 (wall clock): %.2fs" % elapsed)
    kill_pid(pid)
    time.sleep(3)
    return {"cold_start_time": round(elapsed, 2)}

def measure_sleep_wake(sleep_level):
    log("\n===== Sleep Level %d -> Wake 实验 =====" % sleep_level)
    port = 8750 + sleep_level  # 8751 or 8752
    
    # Step 1: 启动 vLLM with sleep mode
    log("Step 1: 启动 vLLM ...")
    pid = start_vllm(port, sleep_mode=True)
    if not wait_ready(port):
        log("ERROR: vLLM 没 ready")
        kill_pid(pid)
        return {"error": "not ready"}
    
    # Step 2: 预热推理
    log("Step 2: 预热推理 ...")
    cmd = build_curl(port, {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    })
    rc, out, err = run_cmd(cmd, timeout=30)
    log("  预热返回: rc=%d len=%d" % (rc, len(out)))
    
    # Step 3: 触发 sleep
    log("Step 3: sleep level %d ..." % sleep_level)
    t_sleep = time.time()
    sleep_payload = {"level": sleep_level}
    cmd_sleep = build_curl(port, sleep_payload, endpoint="/v1/engine/sleep")
    rc, out, err = run_cmd(cmd_sleep, timeout=10)
    sleep_time = time.time() - t_sleep
    log("  Sleep API: rc=%d, 耗时=%.2fs, out=%s" % (rc, sleep_time, out.strip()[:100]))
    
    # 确认 GPU 显存释放
    time.sleep(3)
    rc, out, err = run_cmd("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    log("  Sleep 后 GPU 显存: %s MiB" % out.strip().split("\n")[0])
    
    # Step 4: 触发 wake
    log("Step 4: wake ...")
    t_wake = time.time()
    cmd_wake = build_curl(port, {}, endpoint="/v1/engine/wake")
    rc, out, err = run_cmd(cmd_wake, timeout=60)
    wake_time = time.time() - t_wake
    log("  Wake API: rc=%d, 耗时=%.2fs, out=%s" % (rc, wake_time, out.strip()[:100]))
    
    # Step 5: 验证恢复推理
    log("Step 5: 确认恢复 ...")
    cmd2 = build_curl(port, {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20
    })
    time.sleep(5)
    rc, out, err = run_cmd(cmd2, timeout=30)
    log("  恢复推理: rc=%d, len=%d" % (rc, len(out)))
    
    kill_pid(pid)
    time.sleep(3)
    
    return {
        "sleep_level": sleep_level,
        "sleep_api_time": round(sleep_time, 2),
        "wake_api_time": round(wake_time, 2),
        "sleep_total": round(sleep_time + wake_time, 2)
    }

def main():
    log("=" * 60)
    log("SleepSpot vLLM Sleep Mode 实验")
    log("时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 60)
    
    results = {"model": MODEL_PATH, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    # 清理老 vLLM
    run_cmd("pkill -f 'vllm serve' 2>/dev/null; sleep 2")
    
    # 冷启动
    results["cold_start"] = measure_cold_start()
    
    # Sleep L1
    results["sleep_l1"] = measure_sleep_wake(1)
    
    # Sleep L2
    results["sleep_l2"] = measure_sleep_wake(2)
    
    # 汇总
    log("\n" + "=" * 60)
    log("实验结果汇总")
    log("=" * 60)
    
    cold = results["cold_start"]["cold_start_time"]
    l1w = results["sleep_l1"].get("wake_api_time", "N/A")
    l2w = results["sleep_l2"].get("wake_api_time", "N/A")
    
    log("冷启动时间:     %.2fs (基准)" % cold)
    if isinstance(l1w, (int, float)):
        log("Sleep L1 Wake:  %.2fs (加速 %.1fx)" % (l1w, cold / l1w))
    else:
        log("Sleep L1 Wake:  %s" % l1w)
    if isinstance(l2w, (int, float)):
        log("Sleep L2 Wake:  %.2fs (加速 %.1fx)" % (l2w, cold / l2w))
    else:
        log("Sleep L2 Wake:  %s" % l2w)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log("\nJSON 结果已保存到: %s" % RESULTS_FILE)
    log("=" * 60)

if __name__ == "__main__":
    main()
