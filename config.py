#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 配置模块
从 .env 文件加载所有 LLM 提供商和数据源配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 加载 .env 文件
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


@dataclass
class ProviderConfig:
    """单个 LLM 提供商配置"""
    name: str
    api_key: str
    model: str
    base_url: str
    vision_model: str = ""       # 多模态视觉模型名（空字符串=不支持）
    supports_vision: bool = False  # 推理模型是否直接支持图片输入
    use_proxy: bool = True        # 是否通过 LLM_PROXY 代理（国内服务设 False）
    max_tokens: int = 8192        # 最大输出 token（推理模型需要较大值）
    timeout: float = 300.0        # HTTP/API 请求超时（秒）

    def __repr__(self):
        return f"Provider({self.name}:{self.model})"


@dataclass
class Config:
    """系统总配置"""
    tushare_token: str = ""
    llm_proxy: str = ""
    primary_provider: str = "grok"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    tdx_dir: str = "d:/tdx"   # 本地通达信安装目录

    # 面板大小：每次讨论/辩论调用的LLM数量
    panel_size: int = 4

    # 辩论轮次
    debate_rounds: int = 1

    # 各专家角色的权重
    expert_weights: Dict[str, float] = field(default_factory=lambda: {
        "E1": 0.15,  # 动量换手型
        "E2": 0.20,  # 成长估值型
        "E3": 0.20,  # 多因子平衡型
        "E4": 0.15,  # 技术形态型
        "E5": 0.15,  # 资金流向型
        "E6": 0.15,  # 事件催化型
    })

    def get_available_providers(self) -> List[str]:
        return list(self.providers.keys())

    def get_primary(self) -> Optional[ProviderConfig]:
        if self.primary_provider in self.providers:
            return self.providers[self.primary_provider]
        if self.providers:
            return next(iter(self.providers.values()))
        return None


def load_config() -> Config:
    """加载并返回系统配置"""
    cfg = Config(
        tushare_token=os.getenv("TUSHARE_TOKEN", "").strip(),
        llm_proxy=os.getenv("LLM_PROXY", "").strip(),
        primary_provider=os.getenv("LLM_PROVIDER", "grok").strip(),
        tdx_dir=os.getenv("TDX_DIR", "d:/tdx").strip(),
    )

    # LLM 提供商定义
    provider_specs = [
        {
            "name": "grok",
            "key_env": "GROK_API_KEY",
            "model_env": "GROK_MODEL",
            "default_model": "grok-4-1-fast-reasoning",
            "base_url": "https://api.x.ai/v1",
            "use_proxy": True,
        },
        {
            "name": "gemini",
            "key_env": "GEMINI_API_KEY",
            "model_env": "GEMINI_MODEL",
            "default_model": "gemini-3.1-pro-preview",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "use_proxy": True,
        },
        {
            "name": "deepseek",
            "key_env": "DEEPSEEK_API_KEY",
            "model_env": "DEEPSEEK_MODEL",
            "default_model": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com/v1",
            "use_proxy": False,
        },
        {
            "name": "kimi",
            "key_env": "KIMI_API_KEY",
            "model_env": "KIMI_MODEL",
            "default_model": "kimi-k2.5",
            "base_url": "https://api.moonshot.cn/v1",
            "use_proxy": False,
        },
        {
            "name": "glm",
            "key_env": "GLM_API_KEY",
            "model_env": "GLM_MODEL",
            "default_model": "glm-4-plus",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "use_proxy": False,
        },
        {
            "name": "doubao",
            "key_env": "DOUBAO_API_KEY",
            "model_env": "DOUBAO_MODEL",
            "default_model": "doubao-seed-2-0-pro-260215",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "use_proxy": False,
        },
        {
            "name": "qwen",
            "key_env": "QWEN_API_KEY",
            "model_env": "QWEN_MODEL",
            "default_model": "qwen3.5-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "use_proxy": False,
        },
        {
            "name": "minimax",
            "key_env": "MINIMAX_API_KEY",
            "model_env": "MINIMAX_MODEL",
            "default_model": "MiniMax-M1",
            "base_url": "https://api.minimax.chat/v1",
            "use_proxy": False,
        },
        {
            "name": "perplexity",
            "key_env": "PERPLEXITY_API_KEY",
            "model_env": "PERPLEXITY_MODEL",
            "default_model": "sonar-reasoning-pro",
            "base_url": "https://api.perplexity.ai",
            "use_proxy": True,
        },
    ]

    # 慢速模型默认超时（秒）—— 可通过 {PREFIX}_TIMEOUT 环境变量覆盖
    _default_timeouts = {
        "deepseek": 600.0,
    }

    for spec in provider_specs:
        api_key = os.getenv(spec["key_env"], "").strip()
        if not api_key:
            continue
        model = (os.getenv(spec["model_env"], "") or spec["default_model"]).strip()
        prefix = spec["name"].upper()
        vision_model = os.getenv(f"{prefix}_VISION_MODEL", "").strip()
        supports_vision = (
            os.getenv(f"{prefix}_SUPPORTS_VISION", "false").strip().lower() == "true"
        )
        timeout = float(
            os.getenv(f"{prefix}_TIMEOUT", _default_timeouts.get(spec["name"], 300.0))
        )
        cfg.providers[spec["name"]] = ProviderConfig(
            name=spec["name"],
            api_key=api_key,
            model=model,
            base_url=spec["base_url"],
            vision_model=vision_model,
            supports_vision=supports_vision,
            use_proxy=spec.get("use_proxy", True),
            timeout=timeout,
        )

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(f"主要提供商: {cfg.primary_provider}")
    print(f"可用提供商: {list(cfg.providers.keys())}")
    print(f"代理: {cfg.llm_proxy}")
    print(f"Tushare Token: {'已配置' if cfg.tushare_token else '未配置'}")
