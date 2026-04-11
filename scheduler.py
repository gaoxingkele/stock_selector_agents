#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化调度 — P3 #15

提供两种调度模式：

1. 内置 daemon 模式（开发/测试用）：
     python scheduler.py --daemon
   一直运行，定时触发任务。

2. 单次任务模式（与系统任务计划/cron 集成）：
     python scheduler.py --task pre_market    # 盘前选股
     python scheduler.py --task post_market   # 盘后绩效追踪
     python scheduler.py --task weekly_report # 周日晚周报

任务定义：
  pre_market    08:30 周一-周五  跑链路A+B两条链路，输出当日推荐
  post_market   16:00 周一-周五  绩效追踪+L1/L2缓存清理
  weekly_report 周日 19:00       绩效周报

使用建议:
  Windows: 在"任务计划程序"添加 3 个任务，分别调用对应 --task
  Linux:   crontab 添加 3 行
    30 8  * * 1-5  cd /path && python scheduler.py --task pre_market
    0  16 * * 1-5  cd /path && python scheduler.py --task post_market
    0  19 * * 0    cd /path && python scheduler.py --task weekly_report
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    log_file = LOG_DIR / f"scheduler_{datetime.now():%Y%m%d}.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_trading_day() -> bool:
    """简单判断今日是否为工作日（周一到周五）。
    完整实现应查交易日历，这里足够大多数情况。"""
    return datetime.now().weekday() < 5


def task_pre_market():
    """盘前 08:30：跑链路A+B 两条链路，输出当日推荐"""
    if not is_trading_day():
        _log("非交易日，跳过盘前任务", "INFO")
        return

    _log("开始执行盘前选股（链路 both）", "INFO")
    cmd = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--link", "both",
        "--no-confirm",
        "--incremental",  # 复用已有缓存，避免浪费
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时上限
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            _log("盘前选股完成", "INFO")
        else:
            _log(f"盘前选股失败 (rc={result.returncode}): {result.stderr[:500]}", "ERROR")
    except subprocess.TimeoutExpired:
        _log("盘前选股超时（>1h）", "ERROR")
    except Exception as e:
        _log(f"盘前选股异常: {e}", "ERROR")

    # 推送通知（占位，等待集成微信/邮件）
    try:
        _send_notification("盘前选股已完成", _read_latest_recommendations())
    except Exception as e:
        _log(f"通知发送失败: {e}", "WARN")


def task_post_market():
    """盘后 16:00：绩效追踪 + L1 缓存清理"""
    if not is_trading_day():
        _log("非交易日，跳过盘后任务", "INFO")
        return

    _log("开始执行盘后绩效追踪", "INFO")
    cmd = [sys.executable, str(BASE_DIR / "main.py"), "--perf"]
    try:
        result = subprocess.run(
            cmd, cwd=str(BASE_DIR), capture_output=True, text=True,
            timeout=600, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            _log("绩效追踪完成", "INFO")
        else:
            _log(f"绩效追踪失败: {result.stderr[:500]}", "ERROR")
    except Exception as e:
        _log(f"绩效追踪异常: {e}", "ERROR")

    # 清理超过 7 天的中间缓存（L1/L2/Borda）
    try:
        _cleanup_old_cache(days=7)
    except Exception as e:
        _log(f"缓存清理异常: {e}", "WARN")


def task_weekly_report():
    """周日 19:00：生成绩效周报"""
    _log("开始生成绩效周报", "INFO")

    # 调用 historical_backtest --months 0.25 (1周) 跑近一周回测
    cmd = [
        sys.executable,
        str(BASE_DIR / "historical_backtest.py"),
        "--months", "1",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(BASE_DIR), capture_output=True, text=True,
            timeout=3600, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            _log("周报生成完成", "INFO")
        else:
            _log(f"周报生成失败: {result.stderr[:500]}", "ERROR")
    except Exception as e:
        _log(f"周报异常: {e}", "ERROR")

    # 推送
    try:
        report_path = BASE_DIR / "output" / "backtest_history" / "report.md"
        if report_path.exists():
            content = report_path.read_text(encoding="utf-8")[:3000]
            _send_notification("绩效周报", content)
    except Exception as e:
        _log(f"周报推送失败: {e}", "WARN")


def _read_latest_recommendations() -> str:
    """从最新的 risk_result 文件提取推荐摘要"""
    today = datetime.now().strftime("%Y%m%d")
    risk_file = BASE_DIR / "output" / f"risk_result_{today}.json"
    if not risk_file.exists():
        return "(无当日推荐)"

    import json
    try:
        data = json.loads(risk_file.read_text(encoding="utf-8"))
        approved = data.get("approved", [])
        if not approved:
            return "(本日无推荐)"

        lines = [f"今日推荐 {len(approved)} 只:"]
        for s in approved[:10]:
            code = s.get("code", "")
            name = s.get("name", "")
            score = s.get("final_score", 0)
            pos = s.get("position_advice", "-")
            lines.append(f"  {code} {name} 分数={score:.0f} 仓位={pos}")
        return "\n".join(lines)
    except Exception as e:
        return f"(解析失败: {e})"


def _send_notification(title: str, content: str):
    """推送通知 — 当前为占位实现，写入文件。
    后续可集成: 邮件 (smtplib) / 微信 (wechatpy) / 钉钉 webhook"""
    notif_dir = BASE_DIR / "output" / "notifications"
    notif_dir.mkdir(parents=True, exist_ok=True)
    fname = notif_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{title[:20]}.txt"
    fname.write_text(f"{title}\n{'='*40}\n{content}", encoding="utf-8")
    _log(f"通知已写入: {fname.name}", "INFO")


def _cleanup_old_cache(days: int = 7):
    """清理超过 N 天的中间缓存文件"""
    output_dir = BASE_DIR / "output"
    if not output_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=days)
    cleaned = 0
    for prefix in ("L1_candidates_", "L2_candidates_", "L3_result_",
                   "fusion_result_", "risk_result_", "market_radar_"):
        for f in output_dir.glob(f"{prefix}*.json"):
            try:
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                    f.unlink()
                    cleaned += 1
            except Exception:
                pass
    _log(f"清理了 {cleaned} 个超过 {days} 天的缓存文件", "INFO")


def daemon_loop():
    """内置 daemon：每分钟检查一次时间，触发到点的任务"""
    _log("调度器 daemon 已启动", "INFO")
    last_triggered = {}  # task_name -> date_str (避免重复触发)

    while True:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        weekday = now.weekday()  # 0=周一 6=周日
        hhmm = now.strftime("%H:%M")

        # 盘前任务（周一-周五 08:30）
        if weekday < 5 and hhmm == "08:30" and last_triggered.get("pre_market") != date_str:
            try:
                task_pre_market()
                last_triggered["pre_market"] = date_str
            except Exception as e:
                _log(f"盘前任务异常: {e}", "ERROR")

        # 盘后任务（周一-周五 16:00）
        if weekday < 5 and hhmm == "16:00" and last_triggered.get("post_market") != date_str:
            try:
                task_post_market()
                last_triggered["post_market"] = date_str
            except Exception as e:
                _log(f"盘后任务异常: {e}", "ERROR")

        # 周报任务（周日 19:00）
        if weekday == 6 and hhmm == "19:00" and last_triggered.get("weekly_report") != date_str:
            try:
                task_weekly_report()
                last_triggered["weekly_report"] = date_str
            except Exception as e:
                _log(f"周报任务异常: {e}", "ERROR")

        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="自动化调度器")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--daemon", action="store_true", help="启动内置 daemon 循环")
    group.add_argument(
        "--task",
        choices=["pre_market", "post_market", "weekly_report"],
        help="单次任务模式（与系统任务计划/cron 集成）",
    )
    args = parser.parse_args()

    if args.daemon:
        daemon_loop()
    elif args.task == "pre_market":
        task_pre_market()
    elif args.task == "post_market":
        task_post_market()
    elif args.task == "weekly_report":
        task_weekly_report()


if __name__ == "__main__":
    main()
