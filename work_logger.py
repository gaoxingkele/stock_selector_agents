#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 工作日志模块

JSONL结构化日志 + 控制台输出。

设计：Queue-based 异步写入，支持多任务并发，防止死锁。
  - log() 调用者只做：更新timer dict（加锁，极快）+ queue.put（无锁）
  - 单个后台线程负责所有文件写入（无竞争，无锁）
  - 控制台输出仍在调用方线程完成（即时可见）

事件类型:
    session_start, expert_start, expert_done,
    debate_start, debate_done, model_done,
    fusion_start, fusion_done, error, pipeline_done
"""

import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Optional


class WorkLogger:
    """
    JSONL结构化工作日志，同时输出到控制台。

    线程安全策略：
      - _timers dict  → 由 _lock 保护（极短暂持有）
      - 文件写入      → 唯一后台 writer 线程通过 Queue 完成（无锁，无竞争）
      - 控制台输出    → 调用方线程直接 print（Python print 本身是线程安全的）
    """

    EVENT_ICONS: Dict[str, str] = {
        "session_start":  "[启动]",
        "expert_start":   "[>> ] ",
        "expert_done":    "[OK ] ",
        "debate_start":   "[辩论]",
        "debate_retry":   "[重试]",
        "debate_fallback":"[降级]",
        "debate_done":    "[论毕]",
        "model_done":     "[完成]",
        "fusion_start":   "[融合]",
        "fusion_done":    "[融毕]",
        "error":          "[错误]",
        "pipeline_done":  "[结束]",
    }

    _SENTINEL = object()  # 终止 writer 线程的哨兵对象

    def __init__(self, log_dir: str = "output/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        self.log_path = os.path.join(log_dir, f"work_log_{ts}.log")

        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()  # 仅保护 _timers dict

        # 无界队列：生产者 put_nowait 不阻塞，writer 线程持续消费
        self._queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._file_writer, name="WorkLogWriter", daemon=True
        )
        self._writer.start()

    # ------------------------------------------------------------------ #
    #  后台 writer 线程                                                   #
    # ------------------------------------------------------------------ #

    def _file_writer(self) -> None:
        """
        唯一写文件的线程，从队列中取条目追加到 JSONL 文件。
        文件 append 无需加锁（单线程写入）。
        """
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.task_done()
                break
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass  # 日志写失败不影响主流程
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """
        刷新队列并等待 writer 线程退出（进程正常退出时调用）。
        超时 5s 后强制返回（writer 是 daemon 线程，进程退出时也会终止）。
        """
        self._queue.put(self._SENTINEL)
        self._writer.join(timeout=5)

    # ------------------------------------------------------------------ #
    #  公共接口                                                            #
    # ------------------------------------------------------------------ #

    def log(
        self,
        event: str,
        model: Optional[str] = None,
        detail: Optional[Dict] = None,
    ) -> None:
        """
        写入日志条目并打印到控制台。

        参数:
            event  : 事件类型字符串
            model  : 模型名称（可选）
            detail : 附加信息字典（可选）
        """
        # ── 计时：expert_start / expert_done 配对 ──────────────────────
        timer_key = f"{model}_{(detail or {}).get('expert_id', '')}"
        elapsed: Optional[float] = None

        if event == "expert_start" and (detail or {}).get("expert_id"):
            with self._lock:                  # 持锁极短（仅写一个 float）
                self._timers[timer_key] = time.time()
        elif event in ("expert_done", "error") and (detail or {}).get("expert_id"):
            with self._lock:
                t0 = self._timers.pop(timer_key, None)
            if t0 is not None:
                elapsed = round(time.time() - t0, 2)
                detail = dict(detail or {})
                detail["elapsed"] = elapsed

        # ── 构建条目 ───────────────────────────────────────────────────
        entry: Dict = {"ts": datetime.now().isoformat(), "event": event}
        if model:
            entry["model"] = model
        if detail:
            entry["detail"] = detail

        # ── 异步写文件（put_nowait 不阻塞，不持锁）─────────────────────
        self._queue.put_nowait(entry)

        # ── 控制台输出（调用方线程，即时显示）─────────────────────────
        self._print(event, model, detail)

    # ------------------------------------------------------------------ #
    #  控制台格式化                                                        #
    # ------------------------------------------------------------------ #

    def _print(
        self,
        event: str,
        model: Optional[str],
        detail: Optional[Dict],
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        icon = self.EVENT_ICONS.get(event, "[ ?  ]")
        model_str = f" [{model}]" if model else ""

        extra = ""
        if detail:
            if "expert_id" in detail:
                extra = f" {detail['expert_id']}"
                if "expert_name" in detail:
                    extra += f"({detail['expert_name']})"
            if "picks_count" in detail:
                extra += f" → {detail['picks_count']}只"
            if "elapsed" in detail:
                extra += f" [{detail['elapsed']:.1f}s]"
            if "picks_count" not in detail and "elapsed" not in detail:
                if "message" in detail:
                    extra += f" {detail['message']}"
                elif "error" in detail:
                    extra += f" {str(detail['error'])[:120]}"

        print(f"  {ts} {icon}{model_str}{extra}", flush=True)
