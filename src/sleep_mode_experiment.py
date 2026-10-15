#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SleepSpot 正式实验 (真实 vLLM sleep mode API, RTX 4090)
=====================================================
场景:
  A. Sleep L1 wake x3  - 权重保留在 CPU RAM, wake 后重新映射
  B. Sleep L2 wake x3  - 权重彻底释放, wake 后从磁盘重载
  C. 冷启动 x3         - 全新进程 + 模型加载 (基线; 第1次为真冷启动,
                          第2/3次享受 OS page cache, 即 spot 同机快速重启)
指标 (秒):
  - ready_time:   从启动/spawn 到 /health 200
  - first_infer:  从启动/spawn 到第一次推理完成
  - wake_api:     POST /wake_up 返回耗时
  - mem_restore:  wake 后显存恢复耗时
  - sleep_api:    POST /sleep 返回耗时
输出: /root/sleepspot_results.json
"""

import subprocess, time, json, os, re

MODEL = "/root/models/Qwen2.5-1.5B-Instruct"
PORT = 8890
BASE = "http://localhost:%d" % PORT
VLLM_BIN = "/root/miniconda3/envs/myconda/bin/vllm"
RESULTS = "/root/sleepspot_results.json"
EXPLOG = "/root/sleepspot_exp.log"
SERVER_LOG = "/root/vllm_exp_server.log"

def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(EXPLOG, "a") as f:
        f.write(line + "\n")

def sh(cmd, timeout=120):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)

def gpu_mem_mib():
    r = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    try:
        return int(r.stdout.strip().split("\n")[0].replace(" ", ""))
    except Exception:
        return -1

def wait_gpu_free(threshold=1000, max_s=150):
    t0 = time.time()
    while time.time() - t0 < max_s:
        m = gpu_mem_mib()
        if 0 <= m < threshold:
            return True
        time.sleep(2)
    return False

def health_ok():
    r = sh("curl -s -m 2 %s/health -o /dev/null -w '%%{http_code}'" % BASE,
           timeout=10)
    return r.stdout.strip() == "200"

def wait_ready(max_s=300):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if health_ok():
            return time.time() - t0
        time.sleep(1)
    return None

def post(path, timeout=120):
    """POST, 返回 (http_code, 耗时s)"""
    t0 = time.time()
    r = sh("curl -s -m %d -X POST %s%s -o /dev/null -w '%%{http_code}'"
           % (timeout, BASE, path), timeout=timeout + 15)
    return r.stdout.strip(), time.time() - t0

def get(path, timeout=15):
    r = sh("curl -s -m %d %s%s" % (timeout, BASE, path), timeout=timeout + 10)
    return r.stdout.strip()

def infer(prompt="hello", max_tokens=8, timeout=60):
    """推理一次, 返回 (http_code, 耗时s)"""
    payload = ('{"model":"%s","messages":[{"role":"user","content":"%s"}],'
               '"max_tokens":%d}' % (MODEL, prompt, max_tokens))
    t0 = time.time()
    r = sh("curl -s -m %d %s/v1/chat/completions -H 'Content-Type: application/json' "
           "-d '%s' -o /dev/null -w '%%{http_code}'"
           % (timeout, BASE, payload), timeout=timeout + 15)
    return r.stdout.strip(), time.time() - t0

def kill_all():
    """杀掉所有 vllm 相关进程 + 按 GPU PID 兜底"""
    sh("pkill -9 -f vllm 2>/dev/null")
    sh("pkill -9 -f spawn_main 2>/dev/null")
    r = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    for p in r.stdout.strip().split("\n"):
        p = p.strip().split(",")[0]
        if p.isdigit():
            sh("kill -9 %s 2>/dev/null" % p)
    time.sleep(3)
    ok = wait_gpu_free()
    log("  GPU 清理: %s (现 %d MiB)" % ("OK" if ok else "TIMEOUT", gpu_mem_mib()))
    return ok

def start_server(sleep_mode):
    flag = "--enable-sleep-mode" if sleep_mode else ""
    # PATH 必须含 conda env bin: flashinfer JIT 需要 ninja
    cmd = ("PATH=/root/miniconda3/envs/myconda/bin:/usr/local/bin:/usr/bin:/bin "
           "VLLM_SERVER_DEV_MODE=1 nohup %s serve %s --port %d "
           "--gpu-memory-utilization 0.85 --max-model-len 512 %s "
           "> %s 2>&1 & echo $!" % (VLLM_BIN, MODEL, PORT, flag, SERVER_LOG))
    r = sh(cmd)
    pid = r.stdout.strip().split("\n")[-1]
    log("  server pid=%s sleep_mode=%s" % (pid, sleep_mode))
    return pid

def sleep_wake_trial(level):
    """在已就绪的 server 上: sleep level -> wake, 全程计时"""
    rec = {"level": level}
    code, t = post("/sleep?level=%d" % level, timeout=90)
    rec["sleep_api_code"] = code
    rec["sleep_api"] = round(t, 2)
    time.sleep(2)
    rec["is_sleeping_after"] = get("/is_sleeping")
    rec["gpu_after_sleep_mib"] = gpu_mem_mib()

    t0 = time.time()
    code, t = post("/wake_up", timeout=150)
    rec["wake_code"] = code
    rec["wake_api"] = round(t, 2)

    t0 = time.time()
    while time.time() - t0 < 90:
        if gpu_mem_mib() > 15000:
            break
        time.sleep(0.5)
    rec["mem_restore"] = round(time.time() - t0, 2)
    rec["is_sleeping_after_wake"] = get("/is_sleeping")

    t0 = time.time()
    code, lat = infer("hello", 8, timeout=90)
    rec["wake_total_to_infer"] = round(time.time() - t0, 2)
    rec["infer_code"] = code
    rec["infer_latency"] = round(lat, 2)
    log("  L%d: sleep_api=%.2fs wake_api=%.2fs mem=%.2fs total_to_infer=%.2fs "
        "gpu_after_sleep=%dMiB infer=%s"
        % (level, rec["sleep_api"], rec["wake_api"], rec["mem_restore"],
           rec["wake_total_to_infer"], rec["gpu_after_sleep_mib"], code))
    time.sleep(2)  # 确保空闲再进入下一轮
    return rec

def cold_start_trial(i):
    kill_all()
    log("  冷启动第 %d 次: 启动..." % i)
    t0 = time.time()
    start_server(False)
    ready = wait_ready()
    if ready is None:
        log("  冷启动第 %d 次: NOT READY!" % i)
        tail = sh("tail -5 %s" % SERVER_LOG).stdout
        log("  server log tail: " + tail.replace("\n", " | ")[:300])
        return {"trial": i, "error": "not_ready"}
    t_ready = time.time() - t0
    code, lat = infer("hello", 8, timeout=90)
    t_infer = time.time() - t0
    log("  冷启动第 %d 次: ready=%.2fs infer_code=%s total_to_infer=%.2fs"
        % (i, t_ready, code, t_infer))
    return {"trial": i, "ready_time": round(t_ready, 2),
            "infer_code": code, "total_to_infer": round(t_infer, 2),
            "infer_latency": round(lat, 2)}

def main():
    log("=" * 60)
    log("SleepSpot 正式实验开始 (真实 sleep mode API)")
    log("=" * 60)
    results = {"model": MODEL, "gpu": "RTX 4090 24GB",
               "vllm": "0.8.5 V1",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------- Phase A: sleep/wake trials ----------
    log("Phase A: 准备 sleep mode server")
    kill_all()
    start_server(True)
    ready = wait_ready()
    log("  ready: %.2fs" % ready if ready else "  NOT READY!")
    if ready is None:
        results["error"] = "server not ready"
        json.dump(results, open(RESULTS, "w"), indent=2, ensure_ascii=False)
        return
    code, _ = infer("hi", 8, timeout=90)
    log("  预热: %s" % code)
    time.sleep(2)

    results["sleep_l1"] = []
    results["sleep_l2"] = []
    for i in range(3):
        log("Sleep L1 trial %d" % (i + 1))
        results["sleep_l1"].append(sleep_wake_trial(1))
        log("Sleep L2 trial %d" % (i + 1))
        results["sleep_l2"].append(sleep_wake_trial(2))

    # ---------- Phase B: cold start trials ----------
    results["cold_start"] = []
    for i in range(3):
        log("冷启动 trial %d" % (i + 1))
        results["cold_start"].append(cold_start_trial(i + 1))

    # ---------- 汇总 ----------
    def avg(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 2) if xs else None

    summary = {}
    l1w = [r.get("wake_total_to_infer") for r in results["sleep_l1"]]
    l2w = [r.get("wake_total_to_infer") for r in results["sleep_l2"]]
    l1m = [r.get("mem_restore") for r in results["sleep_l1"]]
    l2m = [r.get("mem_restore") for r in results["sleep_l2"]]
    cs = [r.get("total_to_infer") for r in results["cold_start"]]
    csr = [r.get("ready_time") for r in results["cold_start"]]
    summary["cold_ready_avg"] = avg(csr)
    summary["cold_total_to_infer_avg"] = avg(cs)
    summary["l1_wake_total_avg"] = avg(l1w)
    summary["l1_mem_restore_avg"] = avg(l1m)
    summary["l2_wake_total_avg"] = avg(l2w)
    summary["l2_mem_restore_avg"] = avg(l2m)
    cold = summary["cold_total_to_infer_avg"]
    if cold:
        if summary["l1_wake_total_avg"]:
            summary["l1_speedup"] = round(cold / summary["l1_wake_total_avg"], 1)
        if summary["l2_wake_total_avg"]:
            summary["l2_speedup"] = round(cold / summary["l2_wake_total_avg"], 1)
    results["summary"] = summary

    log("=" * 60)
    log("汇总: " + json.dumps(summary, ensure_ascii=False))
    log("=" * 60)

    # server log 里的 fall asleep 记录
    r = sh("grep -a 'fall asleep' %s" % SERVER_LOG)
    if r.stdout.strip():
        log("server log sleep 记录:\n" + r.stdout.strip())

    json.dump(results, open(RESULTS, "w"), indent=2, ensure_ascii=False)
    log("结果已保存: %s" % RESULTS)
    log("实验结束")

if __name__ == "__main__":
    main()
