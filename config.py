#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 配置模块
从 .env 文件加载所有 LLM 提供商和数据源配置
支持 Cloubic 透明路由：按白名单决定每个 provider 走 Cloubic 还是直连
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 加载 .env 文件（主配置）
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# 加载 .env.cloubic（Cloubic 专用配置，不覆盖已有变量）
_env_cloubic_path = Path(__file__).parent / ".env.cloubic"
if _env_cloubic_path.exists():
    load_dotenv(_env_cloubic_path, override=False)


# ─────────────────────────────────────────────────────────────────────
#  Cloubic 路由配置（模块级变量，运行时可被 --cloubic / --direct 覆盖）
# ─────────────────────────────────────────────────────────────────────

CLOUBIC_ENABLED = os.getenv("CLOUBIC_ENABLED", "false").strip().lower() in ("true", "1", "yes")
CLOUBIC_API_KEY = os.getenv("CLOUBIC_API_KEY", "").strip()
CLOUBIC_BASE_URL = os.getenv("CLOUBIC_BASE_URL", "https://api.cloubic.com/v1").strip()
CLOUBIC_DEFAULT_PROVIDER = os.getenv("CLOUBIC_DEFAULT_PROVIDER", "").strip()

# 白名单：走 Cloubic 路由的 provider（逗号分隔）
# 留空 = 全部走 Cloubic（kimi 除外，始终直连）
CLOUBIC_ROUTED_PROVIDERS = [
    p.strip() for p in os.getenv("CLOUBIC_ROUTED_PROVIDERS", "").split(",") if p.strip()
]

# Cloubic 模型映射：provider → 默认模型
CLOUBIC_MODEL_MAP = {
    "openai":    os.getenv("CLOUBIC_OPENAI_MODEL", "gpt-5.4").strip(),
    "claude":    os.getenv("CLOUBIC_CLAUDE_MODEL", "claude-opus-4-6").strip(),
    "gemini":    os.getenv("CLOUBIC_GEMINI_MODEL", "gemini-3.1-pro-preview").strip(),
    "deepseek":  os.getenv("CLOUBIC_DEEPSEEK_MODEL", "deepseek-v3.2").strip(),
    "grok":      os.getenv("CLOUBIC_GROK_MODEL", "grok-4-1-fast-non-reasoning").strip(),
    "qwen":      os.getenv("CLOUBIC_QWEN_MODEL", "qwen3-max").strip(),
    "doubao":    os.getenv("CLOUBIC_DOUBAO_MODEL", "doubao-seed-1-6-flash-250828").strip(),
    "minimax":   os.getenv("CLOUBIC_MINIMAX_MODEL", "MiniMax-M2.5").strip(),
    # 不在 Cloubic 的 provider，映射到 Cloubic 可用模型作为 fallback
    "kimi":      os.getenv("CLOUBIC_KIMI_MODEL", "deepseek-v3.2").strip(),
    "glm":       os.getenv("CLOUBIC_GLM_MODEL", "glm-5").strip(),
    "perplexity": os.getenv("CLOUBIC_PERPLEXITY_MODEL", "deepseek-v3.2").strip(),
}

# Cloubic 推理模型映射
CLOUBIC_REASONING_MODEL_MAP = {
    "openai":    os.getenv("CLOUBIC_OPENAI_REASONING_MODEL", "gpt-5.2-pro").strip(),
    "claude":    os.getenv("CLOUBIC_CLAUDE_REASONING_MODEL", "claude-sonnet-4-5-20250929-thinking").strip(),
    "gemini":    os.getenv("CLOUBIC_GEMINI_REASONING_MODEL", "gemini-3.1-pro-preview").strip(),
    "deepseek":  os.getenv("CLOUBIC_DEEPSEEK_REASONING_MODEL", "deepSeek-R1-0528").strip(),
    "grok":      os.getenv("CLOUBIC_GROK_REASONING_MODEL", "grok-4-1-fast-reasoning").strip(),
    "qwen":      os.getenv("CLOUBIC_QWEN_REASONING_MODEL", "qwen3-max-2026-01-23").strip(),
    "doubao":    os.getenv("CLOUBIC_DOUBAO_REASONING_MODEL", "doubao-seed-1-6-251015").strip(),
    "minimax":   os.getenv("CLOUBIC_MINIMAX_REASONING_MODEL", "MiniMax-M2.5").strip(),
    "glm":       os.getenv("CLOUBIC_GLM_REASONING_MODEL", "glm-5").strip(),
}


def is_cloubic_mode() -> bool:
    """判断当前是否为 Cloubic 路由模式（运行时动态检查）。"""
    return CLOUBIC_ENABLED and bool(CLOUBIC_API_KEY)


def should_route_via_cloubic(provider: str) -> bool:
    """判断该 provider 是否应走 Cloubic 路由。"""
    if not is_cloubic_mode():
        return False
    if provider == "kimi":  # kimi 始终直连
        return False
    # 白名单为空 = 全部走 Cloubic（kimi 除外）
    if not CLOUBIC_ROUTED_PROVIDERS:
        return True
    return provider in CLOUBIC_ROUTED_PROVIDERS


# ─────────────────────────────────────────────────────────────────────
#  Provider 配置
# ─────────────────────────────────────────────────────────────────────

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

    # 面板大小：每次讨论/辩论调用的LLM数量（默认4: qwen+kimi+openai+claude）
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
    """
    加载并返回系统配置。

    所有直连 provider 从 .env 加载。
    Cloubic 路由在运行时通过 should_route_via_cloubic() 动态决定，
    不影响 provider 列表的加载 — 直连配置始终加载，
    路由决策在 llm_client._get_client() 中透明执行。
    """
    cfg = Config(
        tushare_token=os.getenv("TUSHARE_TOKEN", "").strip(),
        llm_proxy=os.getenv("LLM_PROXY", "").strip(),
        primary_provider=os.getenv("LLM_PROVIDER", "grok").strip(),
        tdx_dir=os.getenv("TDX_DIR", "d:/tdx").strip(),
    )

    # 直连提供商定义
    provider_specs = [
        {
            "name": "grok",
            "key_env": "GROK_API_KEY",
            "model_env": "GROK_MODEL",
            "default_model": "grok-4.20",
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
            "default_model": "glm-5",
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
            "default_model": "MiniMax-M2.7",
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

    # 慢速模型默认超时（秒）
    _default_timeouts = {
        "deepseek": 600.0,
    }

    for spec in provider_specs:
        api_key = os.getenv(spec["key_env"], "").strip()
        # Cloubic 模式下，即使没有直连 API Key，也注册 provider（用 Cloubic Key 替代）
        if not api_key and should_route_via_cloubic(spec["name"]):
            api_key = CLOUBIC_API_KEY  # placeholder，实际调用时用 Cloubic
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

    # Cloubic 模式下，注册白名单中没有直连配置的 Cloubic-only provider
    if is_cloubic_mode():
        for provider_name in CLOUBIC_ROUTED_PROVIDERS:
            if provider_name not in cfg.providers and provider_name in CLOUBIC_MODEL_MAP:
                cloubic_model = CLOUBIC_MODEL_MAP[provider_name]
                cfg.providers[provider_name] = ProviderConfig(
                    name=provider_name,
                    api_key=CLOUBIC_API_KEY,
                    model=cloubic_model,
                    base_url=CLOUBIC_BASE_URL,
                    use_proxy=False,
                    timeout=300.0,
                )

    # Cloubic 模式下覆盖主提供商
    if is_cloubic_mode() and CLOUBIC_DEFAULT_PROVIDER:
        cfg.primary_provider = CLOUBIC_DEFAULT_PROVIDER

    return cfg


# ─────────────────────────────────────────────────────────────────────
#  Cloubic 模型列表查询工具
# ─────────────────────────────────────────────────────────────────────

def list_cloubic_models() -> List[str]:
    """查询 Cloubic API 可用模型列表（需要 CLOUBIC_API_KEY）"""
    try:
        import httpx
        if not CLOUBIC_API_KEY:
            print("  [错误] CLOUBIC_API_KEY 未配置")
            return []
        resp = httpx.get(
            f"{CLOUBIC_BASE_URL}/models",
            headers={"Authorization": f"Bearer {CLOUBIC_API_KEY}"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return sorted(m.get("id", "") for m in data.get("data", []))
        print(f"  [错误] Cloubic 模型查询失败: {resp.status_code}")
        return []
    except Exception as e:
        print(f"  [错误] Cloubic 模型查询异常: {e}")
        return []


def get_connection_mode_str() -> str:
    """返回当前连接模式的可读描述。"""
    if not is_cloubic_mode():
        return "直连 (Direct)"
    if not CLOUBIC_ROUTED_PROVIDERS:
        return "Cloubic 统一路由 (kimi 除外)"
    routed = ",".join(CLOUBIC_ROUTED_PROVIDERS)
    return f"混合 (Cloubic: {routed} | 其他直连)"


if __name__ == "__main__":
    import sys
    # 支持 python config.py [--list-models] 查看配置
    cfg = load_config()
    print(f"\n  连接模式: {get_connection_mode_str()}")
    print(f"  Cloubic 启用: {CLOUBIC_ENABLED}")
    if is_cloubic_mode():
        print(f"  Cloubic 白名单: {CLOUBIC_ROUTED_PROVIDERS or '全部(kimi除外)'}")
    print(f"  主提供商: {cfg.primary_provider}")
    print(f"  已配置提供商: {list(cfg.providers.keys())}")
    print(f"  LLM代理: {cfg.llm_proxy or '未设置'}")
    print(f"  Tushare: {'已配置' if cfg.tushare_token else '未配置'}")
    print(f"\n  各提供商详情:")
    for name, prov in cfg.providers.items():
        via_cloubic = should_route_via_cloubic(name)
        if via_cloubic:
            route_tag = f" [Cloubic → {CLOUBIC_MODEL_MAP.get(name, '?')}]"
        else:
            route_tag = " [直连]" if not prov.use_proxy else " [代理]"
        vision_str = f" | vision={prov.vision_model}" if prov.vision_model else ""
        print(f"    {name}: {prov.model} ({prov.base_url[:40]}...){route_tag}{vision_str}")

    if "--list-models" in sys.argv:
        print("\n  Cloubic 可用模型:")
        for m in list_cloubic_models():
            print(f"    {m}")
