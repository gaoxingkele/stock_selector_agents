#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 多LLM客户端模块

支持提供商: Grok, Gemini, DeepSeek, Kimi, Perplexity, GLM, Doubao
核心功能:
  - 单模型调用
  - 面板多模型并行调用
  - 多轮辩论机制（各模型看到彼此结论后再辩论）
  - JSON结构化输出解析
  - 多模型投票聚合
"""

import base64
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from config import (
    Config, ProviderConfig,
    should_route_via_cloubic, is_cloubic_mode,
    CLOUBIC_API_KEY, CLOUBIC_BASE_URL, CLOUBIC_MODEL_MAP,
)


# 提供商优先级（面板选择顺序）
# 默认4个: qwen(直连) + kimi(直连) + openai/gpt-5.4(Cloubic) + claude/opus-4-6(Cloubic)
PANEL_PRIORITY = [
    "qwen", "kimi", "openai", "claude",
    "grok", "doubao", "gemini", "glm", "deepseek",
    "minimax", "perplexity",
]


# ===================================================================== #
#  LLM调用日志（独立 JSONL 文件，Queue-based 异步写入）                #
# ===================================================================== #

class _LLMCallLogger:
    """
    轻量异步日志器，将每次 LLM API 调用写入独立 JSONL 文件。
    使用单后台线程写文件，主调用线程 put_nowait 不阻塞。
    """

    _SENTINEL = object()

    def __init__(self, log_dir: str = "output/logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d")
        self.log_path = os.path.join(log_dir, f"llm_calls_{ts}.jsonl")
        self._queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._run, name="LLMCallLogger", daemon=True
        )
        self._writer.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.task_done()
                break
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def write(self, entry: Dict) -> None:
        self._queue.put_nowait(entry)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._writer.join(timeout=3)


# 模块级单例，所有 LLMClient 实例共享一个日志文件
_llm_call_logger = _LLMCallLogger()


# ===================================================================== #
#  ConversationSession                                                   #
# ===================================================================== #

class ConversationSession:
    """
    与单个 LLM 模型的连续对话会话，维护完整消息历史。

    用于 ModelTask 中同一模型顺序执行各专家分析的场景：
    每次 say() 调用都会把 user/assistant 消息追加到 self.messages，
    保证模型能看到完整的对话上下文。
    """

    def __init__(self, client: "LLMClient", provider_name: str, system_prompt: str,
                 max_session_time: float = 720.0, max_calls: int = 25):
        self.messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        self._client = client
        self._provider = provider_name
        self.max_session_time = max_session_time
        self.max_calls = max_calls
        self._session_start = time.time()
        self._call_count = 0

    def say(self, content: str, images: Optional[List[str]] = None) -> Optional[str]:
        """
        添加 user 消息（含可选图片），调用 LLM，追加 assistant 消息，返回响应文本。
        若 LLM 调用失败（超时等），自动移除孤立的 user 消息，避免连续 user 消息干扰后续调用。

        参数:
            content : 文本内容
            images  : PNG/JPG 文件路径列表，自动 base64 编码插入 content block
        """
        elapsed = time.time() - self._session_start
        if elapsed > self.max_session_time:
            print(f"  [警告] [{self._provider}] 会话已超时 ({elapsed:.0f}s > {self.max_session_time:.0f}s)，跳过调用")
            return None
        self._call_count += 1
        if self._call_count > self.max_calls:
            print(f"  [警告] [{self._provider}] 会话调用次数已达上限 ({self._call_count} > {self.max_calls})，跳过调用")
            return None

        if images:
            content_blocks: List[Dict] = [{"type": "text", "text": content}]
            for img_path in images:
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    })
                except Exception as e:
                    print(f"  [警告] 图片加载失败 {img_path}: {e}")
            self.messages.append({"role": "user", "content": content_blocks})
        else:
            self.messages.append({"role": "user", "content": content})

        resp = self._client.call(
            provider_name=self._provider,
            messages=self.messages,
        )
        if resp:
            self.messages.append({"role": "assistant", "content": resp})
        else:
            # 调用失败：移除刚追加的 user 消息，防止连续 user 消息干扰后续专家
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
        return resp

    def compact_last_assistant(self) -> None:
        """
        压缩最后一条 assistant 消息：仅保留 JSON 结果，丢弃冗长的分析文本。
        在每位专家完成后调用，大幅减少后续专家面对的上下文长度。
        """
        if len(self.messages) < 2:
            return
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content", "")
        if not content or len(content) <= 600:
            return  # 已经足够精简

        parsed = LLMClient.parse_json(content)
        if parsed:
            compact = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            self.messages[-1]["content"] = compact
        elif len(content) > 1200:
            # 无法提取 JSON，截断保留关键信息
            self.messages[-1]["content"] = content[:800] + "\n...(已截断)"

    def say_with_vision(
        self,
        content: str,
        image_paths: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        带视觉能力的消息发送：
          - 若 provider.supports_vision=True  → 图片直接嵌入 content block
          - 若 provider.supports_vision=False 且有 vision_model → 先用视觉模型获取图表描述，
            将描述文本追加到 content，再调用主推理模型
          - 无图片或无视觉能力              → 等同于 say(content)
        """
        elapsed = time.time() - self._session_start
        if elapsed > self.max_session_time:
            print(f"  [警告] [{self._provider}] 会话已超时 ({elapsed:.0f}s > {self.max_session_time:.0f}s)，跳过调用")
            return None
        self._call_count += 1
        if self._call_count > self.max_calls:
            print(f"  [警告] [{self._provider}] 会话调用次数已达上限 ({self._call_count} > {self.max_calls})，跳过调用")
            return None

        if not image_paths:
            return self.say(content)

        prov: Optional[ProviderConfig] = self._client.config.providers.get(self._provider)
        if not prov:
            return self.say(content)

        if prov.supports_vision:
            # 主模型直接支持图片
            return self.say(content, images=image_paths)

        if prov.vision_model:
            # 用视觉模型先获取图表描述，再注入文本
            desc_parts: List[str] = []
            for img_path in image_paths:
                desc = self._client.call_with_image(
                    provider_name=self._provider,
                    image_path=img_path,
                    question="请详细描述这张图表，包括趋势方向、关键指标数值和重要技术信号。",
                )
                if desc:
                    desc_parts.append(f"[图表分析]\n{desc}")

            enriched = (
                content + "\n\n" + "\n\n".join(desc_parts)
                if desc_parts
                else content
            )
            return self.say(enriched)

        # 无任何视觉能力：回落到 grok 视觉模型获取图表描述
        grok_prov = self._client.config.providers.get("grok")
        if grok_prov and (grok_prov.supports_vision or grok_prov.vision_model):
            desc_parts: List[str] = []
            for img_path in image_paths:
                desc = self._client.call_with_image(
                    provider_name="grok",
                    image_path=img_path,
                    question="请详细描述这张K线图表，包括趋势方向、关键价格区间、均线状态和重要技术信号。",
                )
                if desc:
                    desc_parts.append(f"[图表分析（Grok视觉）]\n{desc}")
            if desc_parts:
                enriched = content + "\n\n" + "\n\n".join(desc_parts)
                return self.say(enriched)

        # grok 也不可用，忽略图片
        return self.say(content)


class LLMClient:
    """多LLM统一客户端，支持投票和辩论机制"""

    def __init__(self, config: Config):
        self.config = config
        self.proxy = config.llm_proxy
        self._clients: Dict[str, OpenAI] = {}
        # 全局模型约束：若设置，get_panel() 只从此列表中选取
        self._active_models: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  内部工具                                                            #
    # ------------------------------------------------------------------ #

    def _get_client(self, provider_name: str) -> OpenAI:
        """
        懒初始化 OpenAI 客户端。
        Cloubic 路由: 白名单内的 provider 走 Cloubic（国内直连，无需代理）
        直连模式: 国外LLM走代理，国内LLM直连（trust_env=False 屏蔽系统代理）
        """
        # Cloubic 路由的 provider 使用独立 key 缓存
        via_cloubic = should_route_via_cloubic(provider_name)
        cache_key = f"{provider_name}__cloubic" if via_cloubic else provider_name

        if cache_key not in self._clients:
            prov: ProviderConfig = self.config.providers[provider_name]
            timeout = getattr(prov, "timeout", 120.0)

            if via_cloubic:
                # Cloubic 路由：统一 API Key + Base URL，国内直连无代理
                http_client = httpx.Client(timeout=timeout, trust_env=False)
                self._clients[cache_key] = OpenAI(
                    api_key=CLOUBIC_API_KEY,
                    base_url=CLOUBIC_BASE_URL,
                    http_client=http_client,
                    timeout=timeout,
                    max_retries=0,
                )
            else:
                # 直连模式
                use_proxy = getattr(prov, "use_proxy", True)
                if use_proxy and self.proxy:
                    http_client = httpx.Client(proxy=self.proxy, timeout=timeout)
                else:
                    http_client = httpx.Client(timeout=timeout, trust_env=False)
                self._clients[cache_key] = OpenAI(
                    api_key=prov.api_key,
                    base_url=prov.base_url,
                    http_client=http_client,
                    timeout=timeout,
                    max_retries=0,
                )
        return self._clients[cache_key]

    def _close(self):
        """关闭所有 HTTP 客户端"""
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()

    # ------------------------------------------------------------------ #
    #  单模型调用                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _input_stats(messages: List[Dict]) -> Dict:
        """统计输入消息的字符数和最后一条用户消息摘要"""
        total_chars = 0
        last_user = ""
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                # vision content blocks
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            total_chars += len(str(c))
            if m.get("role") == "user":
                last_user = str(c)
        return {
            "msgs": len(messages),
            "chars": total_chars,
            "last_user_snippet": last_user[-200:] if last_user else "",
        }

    def call(
        self,
        provider_name: str,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 8192,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        调用指定 LLM 提供商，返回响应文本。
        国外LLM (use_proxy=True): 走代理
        国内LLM (use_proxy=False): 直连，trust_env=False 屏蔽系统代理
        429/超时/HTML错误不重试。
        """
        if provider_name not in self.config.providers:
            print(f"  [警告] 提供商 {provider_name!r} 未配置，跳过")
            return None

        prov: ProviderConfig = self.config.providers[provider_name]
        client = self._get_client(provider_name)
        stats = self._input_stats(messages)

        # Cloubic 路由时使用映射模型名
        via_cloubic = should_route_via_cloubic(provider_name)
        actual_model = CLOUBIC_MODEL_MAP.get(provider_name, prov.model) if via_cloubic else prov.model
        route_tag = "[C]" if via_cloubic else ""

        for attempt in range(max_retries + 1):
            attempt_tag = f"(重试{attempt})" if attempt > 0 else ""
            model_short = actual_model[:40]
            try:
                print(
                    f"  [{provider_name}]{route_tag}{attempt_tag} > {model_short}"
                    f" ({stats['msgs']}msg/{stats['chars']}ch)...",
                    end="", flush=True,
                )
            except UnicodeEncodeError:
                print(
                    f"  [{provider_name}]{route_tag}{attempt_tag} > {model_short}...",
                    end="", flush=True,
                )
            t_call = time.time()
            log_entry: Dict = {
                "ts": datetime.now().isoformat(),
                "provider": provider_name,
                "model": actual_model,
                "via_cloubic": via_cloubic,
                "attempt": attempt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "input_msgs": stats["msgs"],
                "input_chars": stats["chars"],
                "last_user_snippet": stats["last_user_snippet"],
            }
            try:
                kwargs: Dict[str, Any] = {
                    "model": actual_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                if provider_name == "kimi" and "k2" in prov.model:
                    kwargs["temperature"] = 1.0   # kimi-k2系列仅支持temperature=1
                elif provider_name != "perplexity":
                    kwargs["temperature"] = temperature

                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content
                # 推理模型（deepseek-reasoner / kimi-k2等）可能把结果放在 reasoning_content
                if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    content = msg.reasoning_content
                elapsed = round(time.time() - t_call, 2)
                _len = len(content.strip()) if content else 0

                usage = getattr(resp, "usage", None)
                if usage:
                    log_entry["prompt_tokens"]     = getattr(usage, "prompt_tokens", None)
                    log_entry["completion_tokens"] = getattr(usage, "completion_tokens", None)
                    log_entry["total_tokens"]      = getattr(usage, "total_tokens", None)

                log_entry.update({"status": "ok", "elapsed": elapsed, "output_chars": _len})
                _llm_call_logger.write(log_entry)
                print(f" OK {elapsed:.1f}s {_len}ch")
                return content.strip() if content else None

            except Exception as exc:
                elapsed = round(time.time() - t_call, 2)
                err_str = str(exc)
                log_entry.update({
                    "status": "fail",
                    "elapsed": elapsed,
                    "error": err_str[:300],
                })
                _llm_call_logger.write(log_entry)

                # 429限流/超时/HTML错误页：重试无意义，立即放弃
                no_retry = (
                    "429" in err_str
                    or "timed out" in err_str.lower()
                    or "timeout" in err_str.lower()
                    or err_str.lstrip().startswith("<")
                )
                if no_retry or attempt >= max_retries:
                    tag = ""
                    if "429" in err_str:
                        tag = "（限流，跳过重试）"
                    elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
                        prov_timeout = getattr(prov, "timeout", 120.0)
                        tag = f"（超时>{prov_timeout:.0f}s，跳过重试）"
                    elif err_str.lstrip().startswith("<"):
                        tag = "（HTML错误页，跳过重试）"
                    print(f" FAIL {elapsed:.1f}s {tag}: {err_str[:120]}")
                    return None

                wait = 2 ** attempt
                print(f" FAIL {elapsed:.1f}s ({err_str[:80]}), wait {wait}s")
                time.sleep(wait)

    def call_primary(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> Optional[str]:
        """调用主要提供商，失败时依次尝试其他可用提供商，每次切换打印日志"""
        primary = self.config.primary_provider
        result = self.call(primary, messages, temperature, max_tokens)
        if result:
            return result

        # 依次尝试其他提供商（fallback）
        for name in PANEL_PRIORITY:
            if name != primary and name in self.config.providers:
                print(f"  [call_primary] 主提供商 {primary!r} 失败，切换至 {name!r}", flush=True)
                _llm_call_logger.write({
                    "ts": datetime.now().isoformat(),
                    "event": "fallback",
                    "from": primary,
                    "to": name,
                })
                result = self.call(name, messages, temperature, max_tokens)
                if result:
                    return result
        return None

    def call_with_image(
        self,
        provider_name: str,
        image_path: str,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """
        使用 provider 的 vision_model 分析图片，返回描述文本。
        若 vision_model 未配置，返回 None。
        """
        prov: Optional[ProviderConfig] = self.config.providers.get(provider_name)
        if not prov or not prov.vision_model:
            return None

        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"  [警告] 读取图片失败 {image_path}: {e}")
            return None

        client = self._get_client(provider_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            resp = client.chat.completions.create(
                model=prov.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"  [警告] 视觉模型调用失败 [{provider_name}/{prov.vision_model}]: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  面板多模型调用                                                      #
    # ------------------------------------------------------------------ #

    def set_active_models(self, models: List[str]) -> None:
        """设置全局模型约束，后续所有 get_panel/call_panel 只从此列表选取。"""
        self._active_models = [m for m in models if m in self.config.providers]

    def get_panel(self, max_models: Optional[int] = None) -> List[str]:
        """
        按优先级返回面板提供商列表。
        若 _active_models 已设置，只从中选取（尊重 --models 参数）。
        max_models 默认使用 config.panel_size。
        """
        n = max_models if max_models is not None else self.config.panel_size

        if self._active_models:
            # 用户指定了模型列表，按指定顺序返回
            return self._active_models[:n]

        ordered = [p for p in PANEL_PRIORITY if p in self.config.providers]
        # 补充不在优先级列表中的提供商
        for p in self.config.providers:
            if p not in ordered:
                ordered.append(p)
        return ordered[:n]

    def call_panel(
        self,
        messages: List[Dict],
        max_models: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        verbose: bool = True,
        models: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        并行(顺序)调用多个LLM，返回 {provider_name: response_text}。
        models: 指定模型列表（优先级高于 max_models）
        """
        if models:
            panel = [m for m in models if m in self.config.providers]
        else:
            panel = self.get_panel(max_models)
        results: Dict[str, str] = {}

        for name in panel:
            prov = self.config.providers[name]
            if verbose:
                print(f"    [{name} / {prov.model}] 分析中...")
            resp = self.call(name, messages, temperature, max_tokens)
            if resp:
                results[name] = resp
            elif verbose:
                print(f"    [{name}] 未返回结果，跳过")

        return results

    # ------------------------------------------------------------------ #
    #  多轮辩论机制                                                        #
    # ------------------------------------------------------------------ #

    def run_debate(
        self,
        build_prompt_fn: Callable[[str], List[Dict]],
        summarize_fn: Callable[[Dict[str, str]], str],
        max_models: Optional[int] = None,
        rounds: int = 1,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        verbose: bool = True,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        多轮辩论流程:

        Round 1 — 各模型独立分析，输出 JSON 结果。
        Round N — 每模型看到其他模型结论后提出质疑或认可，更新选择。
        返回: {
            "round1": {provider: response},
            "final":  {provider: response},
        }

        参数:
            build_prompt_fn(provider_name) -> messages  构建初始提示词
            summarize_fn(results_dict)     -> str       将上一轮结果合并为可供辩论的文本
            models: 指定模型列表（优先级高于 max_models）
        """
        if models:
            panel = [m for m in models if m in self.config.providers]
        else:
            panel = self.get_panel(max_models)

        # ---------- Round 1: 独立分析 ----------
        if verbose:
            print(f"\n  ── 第1轮：独立分析（{len(panel)}个模型）──")

        round1: Dict[str, str] = {}
        for name in panel:
            prov = self.config.providers[name]
            if verbose:
                print(f"    [{name}/{prov.model}] 独立分析...")
            msgs = build_prompt_fn(name)
            resp = self.call(name, msgs, temperature, max_tokens)
            if resp:
                round1[name] = resp
            elif verbose:
                print(f"    [{name}] 未返回结果")

        if not round1:
            return {"round1": {}, "final": {}}

        current = dict(round1)

        # ---------- 辩论轮次 ----------
        for rnd in range(rounds):
            if verbose:
                print(f"\n  ── 第{rnd+2}轮：辩论审视 ──")

            debate_ctx = summarize_fn(current)
            next_round: Dict[str, str] = {}

            for name in panel:
                if name not in current:
                    continue
                prov = self.config.providers[name]
                if verbose:
                    print(f"    [{name}/{prov.model}] 审视并辩论...")

                # 在原始对话基础上注入辩论上下文
                msgs = build_prompt_fn(name)
                msgs.append({"role": "assistant", "content": current[name]})
                msgs.append({
                    "role": "user",
                    "content": (
                        f"【其他AI模型的分析结果汇总如下】\n\n{debate_ctx}\n\n"
                        "---\n"
                        "请你基于自己的专业判断，仔细审视上述其他模型的观点：\n"
                        "1. 明确说明你同意哪些观点及理由（支持）\n"
                        "2. 明确说明你质疑哪些观点及理由（反对）\n"
                        "3. 给出你的最终选股决策（可维持或调整原有选择）\n\n"
                        "⚠️ 请仍以 JSON 格式返回最终决策。"
                    ),
                })

                resp = self.call(name, msgs, temperature, max_tokens)
                if resp:
                    next_round[name] = resp

            if next_round:
                current = next_round

        return {"round1": round1, "final": current}

    # ------------------------------------------------------------------ #
    #  JSON 解析 & 结果聚合                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_json(text: str) -> Optional[Dict]:
        """从 LLM 响应文本中提取 JSON 对象"""
        if not text:
            return None

        # 1) 从 markdown 代码块提取
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            for match in re.findall(pattern, text):
                try:
                    return json.loads(match.strip())
                except Exception:
                    continue

        # 2) 直接解析
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # 3) 提取最外层 {} 块
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if 0 <= start < end:
                return json.loads(text[start:end])
        except Exception:
            pass

        return None

    @staticmethod
    def aggregate_picks(
        results: Dict[str, str],
    ) -> Dict[str, Dict]:
        """
        聚合多个模型的选股结果（JSON 中含 picks 数组）。

        返回: {code: {code, name, sector, votes, avg_score, avg_stars,
                       consensus_score, voters, reasonings}}
        """
        votes: Dict[str, Dict] = {}

        for provider_name, text in results.items():
            parsed = LLMClient.parse_json(text)
            if not parsed:
                continue

            picks = parsed.get("picks", [])
            if not isinstance(picks, list):
                continue

            for pick in picks:
                if not isinstance(pick, dict):
                    continue

                code = str(pick.get("code", "")).strip().lstrip("0") or str(pick.get("code", "")).strip()
                # 保留原始code格式
                code = str(pick.get("code", "")).strip()
                if not code:
                    continue

                if code not in votes:
                    votes[code] = {
                        "code": code,
                        "name": pick.get("name", ""),
                        "sector": pick.get("sector", ""),
                        "votes": 0,
                        "total_score": 0,
                        "total_stars": 0,
                        "voters": [],
                        "reasonings": {},
                        "pick_details": {},
                    }

                score = float(pick.get("score", 70))
                stars = float(pick.get("stars", 3))
                votes[code]["votes"] += 1
                votes[code]["total_score"] += score
                votes[code]["total_stars"] += stars
                votes[code]["voters"].append(provider_name)
                votes[code]["reasonings"][provider_name] = pick.get("reasoning", "")
                votes[code]["pick_details"][provider_name] = pick

                # 补全名称/板块
                if pick.get("name") and not votes[code]["name"]:
                    votes[code]["name"] = pick["name"]
                if pick.get("sector") and not votes[code]["sector"]:
                    votes[code]["sector"] = pick["sector"]

        # 计算平均值和共识分
        for code, data in votes.items():
            n = data["votes"]
            data["avg_score"] = round(data["total_score"] / n, 1) if n else 0
            data["avg_stars"] = round(data["total_stars"] / n, 1) if n else 0
            # 共识奖励：每多1票加5分
            data["consensus_score"] = data["avg_score"] + max(0, (n - 1)) * 5

        return votes

    @staticmethod
    def build_debate_summary(results: Dict[str, str]) -> str:
        """将多模型结果构建为可读的辩论上下文文本"""
        lines = []
        for provider, text in results.items():
            parsed = LLMClient.parse_json(text)
            if parsed and "picks" in parsed:
                picks_info = []
                for p in parsed["picks"]:
                    code = p.get("code", "?")
                    name = p.get("name", "")
                    score = p.get("score", "?")
                    stars = p.get("stars", "?")
                    reasoning = p.get("reasoning", "")[:80]
                    picks_info.append(
                        f"  • {code}{' ' + name if name else ''}"
                        f"（{stars}星/{score}分）: {reasoning}"
                    )
                lines.append(
                    f"【{provider}】的选择：\n" + "\n".join(picks_info)
                    if picks_info
                    else f"【{provider}】未给出有效选择"
                )
            else:
                # 原始文本摘要
                summary = text[:300].replace("\n", " ")
                lines.append(f"【{provider}】: {summary}...")

        return "\n\n".join(lines)
