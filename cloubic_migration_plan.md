# Cloubic API 迁移方案

## 背景

将现有9个LLM提供商（grok/gemini/deepseek/kimi/glm/doubao/qwen/minimax/perplexity）的直连访问，统一替换为通过 Cloubic API 网关中转。

## Cloubic API 关键信息

| 项目 | 值 |
|------|-----|
| Base URL | `https://api.cloubic.com/v1` |
| 认证方式 | `Authorization: Bearer <CLOUBIC_API_KEY>` |
| 协议兼容 | 完全兼容 OpenAI SDK（`/v1/chat/completions`） |
| 流式传输 | 支持 SSE |
| Tool Calling | 支持 |
| 并发限制 | 文档未明确，需实测 |

## 迁移优势

1. **统一 API Key** — 不再需要维护9个提供商各自的密钥
2. **无需代理** — Cloubic 国内可直连，省去 `LLM_PROXY` 和 `use_proxy` 逻辑
3. **简化客户端** — 所有模型共享同一 `base_url`，可复用 OpenAI client 实例
4. **真正并行** — 统一网关下，可安全地将 `call_panel()` 改为多线程并行调用

## 待确认事项

- [ ] 获取 Cloubic API Key 并确认额度
- [ ] 在 Cloubic 模型市场确认各模型ID是否可用（尤其国产模型 kimi/glm/doubao/qwen/minimax）
- [ ] 确认 Cloubic 对 vision 模型的支持情况（图片 base64 输入）
- [ ] 确认 Cloubic 并发上限（当前需要最多9路并行）
- [ ] 决定是否保留原直连方式作为 fallback

---

## 修改计划

### 1. `.env` 文件改动

**新增：**
```env
# Cloubic 统一网关
CLOUBIC_API_KEY=你的cloubic密钥
CLOUBIC_BASE_URL=https://api.cloubic.com/v1
```

**简化：** 各提供商不再需要单独的 `API_KEY` 和 `BASE_URL`，只保留 `MODEL` 配置：
```env
# 只需保留模型名（API Key 和 Base URL 统一走 Cloubic）
GROK_MODEL=grok-4.20-beta-0309-reasoning
GEMINI_MODEL=gemini-3.1-pro-preview
DEEPSEEK_MODEL=deepseek-reasoner
KIMI_MODEL=kimi-k2.5
GLM_MODEL=glm-4.7
DOUBAO_MODEL=doubao-seed-2-0-pro-260215
QWEN_MODEL=qwen3.5-plus
MINIMAX_MODEL=MiniMax-M1
PERPLEXITY_MODEL=sonar-pro

# Vision 模型配置仍保留
GROK_VISION_MODEL=grok-4.20-beta-0309-reasoning
GROK_SUPPORTS_VISION=true
# ... 其他 vision 配置同理
```

**可删除：**
- `LLM_PROXY`（不再需要代理）
- 各提供商的 `*_API_KEY`（统一用 `CLOUBIC_API_KEY`）

### 2. `config.py` 改动

#### 2.1 ProviderConfig 精简

```python
@dataclass
class ProviderConfig:
    name: str
    api_key: str          # 统一为 CLOUBIC_API_KEY
    model: str
    base_url: str         # 统一为 CLOUBIC_BASE_URL
    vision_model: str = ""
    supports_vision: bool = False
    # use_proxy: bool = True    ← 删除，不再需要
    max_tokens: int = 8192
    timeout: float = 300.0
```

#### 2.2 load_config() 改动

```python
def load_config() -> Config:
    cfg = Config(
        tushare_token=os.getenv("TUSHARE_TOKEN", "").strip(),
        # llm_proxy 不再需要
        primary_provider=os.getenv("LLM_PROVIDER", "grok").strip(),
    )

    # Cloubic 统一配置
    cloubic_key = os.getenv("CLOUBIC_API_KEY", "").strip()
    cloubic_url = os.getenv("CLOUBIC_BASE_URL", "https://api.cloubic.com/v1").strip()

    if not cloubic_key:
        print("[错误] CLOUBIC_API_KEY 未配置")
        return cfg

    # 各提供商只需定义 name 和 default_model
    provider_specs = [
        {"name": "grok",      "model_env": "GROK_MODEL",      "default_model": "grok-4.20-beta-0309-reasoning"},
        {"name": "gemini",    "model_env": "GEMINI_MODEL",     "default_model": "gemini-3.1-pro-preview"},
        {"name": "deepseek",  "model_env": "DEEPSEEK_MODEL",   "default_model": "deepseek-reasoner"},
        {"name": "kimi",      "model_env": "KIMI_MODEL",       "default_model": "kimi-k2.5"},
        {"name": "glm",       "model_env": "GLM_MODEL",        "default_model": "glm-4.7"},
        {"name": "doubao",    "model_env": "DOUBAO_MODEL",     "default_model": "doubao-seed-2-0-pro-260215"},
        {"name": "qwen",      "model_env": "QWEN_MODEL",       "default_model": "qwen3.5-plus"},
        {"name": "minimax",   "model_env": "MINIMAX_MODEL",    "default_model": "MiniMax-M1"},
        {"name": "perplexity","model_env": "PERPLEXITY_MODEL",  "default_model": "sonar-pro"},
    ]

    for spec in provider_specs:
        model = (os.getenv(spec["model_env"], "") or spec["default_model"]).strip()
        prefix = spec["name"].upper()
        vision_model = os.getenv(f"{prefix}_VISION_MODEL", "").strip()
        supports_vision = os.getenv(f"{prefix}_SUPPORTS_VISION", "false").strip().lower() == "true"
        timeout = float(os.getenv(f"{prefix}_TIMEOUT", 300.0))

        cfg.providers[spec["name"]] = ProviderConfig(
            name=spec["name"],
            api_key=cloubic_key,        # 统一 key
            model=model,
            base_url=cloubic_url,       # 统一 URL
            vision_model=vision_model,
            supports_vision=supports_vision,
            timeout=timeout,
        )

    return cfg
```

### 3. `llm_client.py` 改动

#### 3.1 `_get_client()` 简化

不再需要代理判断逻辑，所有 provider 直连 Cloubic：

```python
def _get_client(self, provider_name: str) -> OpenAI:
    if provider_name not in self._clients:
        prov = self.config.providers[provider_name]
        timeout = getattr(prov, "timeout", 300.0)
        # Cloubic 国内直连，不需要代理，trust_env=False 避免系统代理干扰
        http_client = httpx.Client(timeout=timeout, trust_env=False)
        self._clients[provider_name] = OpenAI(
            api_key=prov.api_key,
            base_url=prov.base_url,
            http_client=http_client,
            timeout=timeout,
            max_retries=0,
        )
    return self._clients[provider_name]
```

**进一步优化（可选）：** 由于所有 provider 共享同一 `api_key` + `base_url`，可以只创建一个 OpenAI client 实例复用：

```python
def _get_client(self, provider_name: str) -> OpenAI:
    if "_shared" not in self._clients:
        prov = next(iter(self.config.providers.values()))
        http_client = httpx.Client(timeout=300.0, trust_env=False)
        self._clients["_shared"] = OpenAI(
            api_key=prov.api_key,
            base_url=prov.base_url,
            http_client=http_client,
            timeout=300.0,
            max_retries=0,
        )
    return self._clients["_shared"]
```

> 注意：如果不同 provider 需要不同 timeout，则仍需按 provider 创建独立 client。

#### 3.2 删除代理相关代码

- 删除 `self.proxy = config.llm_proxy`
- 删除 `_get_client()` 中的 `use_proxy` 判断分支
- Config 类中删除 `llm_proxy` 字段

#### 3.3 `call_panel()` 改为真正并行（可选但推荐）

当前 `call_panel()` 是顺序调用（for 循环逐个调），迁移到 Cloubic 后所有请求走同一网关，可安全并行：

```python
import concurrent.futures

def call_panel(
    self,
    messages: List[Dict],
    max_models: Optional[int] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    verbose: bool = True,
) -> Dict[str, str]:
    panel = self.get_panel(max_models)
    results: Dict[str, str] = {}

    def _call_one(name: str) -> Tuple[str, Optional[str]]:
        if verbose:
            prov = self.config.providers[name]
            print(f"    [{name} / {prov.model}] 分析中...")
        resp = self.call(name, messages, temperature, max_tokens)
        return name, resp

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(panel)) as executor:
        futures = {executor.submit(_call_one, name): name for name in panel}
        for future in concurrent.futures.as_completed(futures):
            name, resp = future.result()
            if resp:
                results[name] = resp
            elif verbose:
                print(f"    [{name}] 未返回结果，跳过")

    return results
```

#### 3.4 `run_debate()` 第一轮也可并行

辩论的 Round 1（独立分析）同理可改为并行，Round 2+（辩论轮）因依赖上一轮结果仍需顺序。

### 4. 其他文件

- **`stock_agents.py`** — 无需改动（通过 `LLMClient` 间接调用）
- **`main.py`** — 无需改动（通过 `LLMClient` 间接调用）
- **`data_engine.py`** — 无需改动（数据采集不走 LLM）
- **`report_generator.py`** — 无需改动

---

## 模型ID映射（待确认）

Cloubic 文档示例中使用的模型ID：
- `gpt-4o` / `gpt-5.1` / `gpt-5-pro`
- `claude-3-5-sonnet-20241022`
- `gemini-2.0-flash`

当前项目使用的模型ID需逐一确认 Cloubic 是否支持：

| 提供商 | 当前模型ID | Cloubic 是否支持 | 备选模型ID |
|--------|-----------|-----------------|-----------|
| grok | grok-4.20-beta-0309-reasoning | ? | — |
| gemini | gemini-3.1-pro-preview | ? | gemini-2.0-flash |
| deepseek | deepseek-reasoner | ? | deepseek-chat |
| kimi | kimi-k2.5 | ? | — |
| glm | glm-4.7 | ? | — |
| doubao | doubao-seed-2-0-pro-260215 | ? | — |
| qwen | qwen3.5-plus | ? | — |
| minimax | MiniMax-M1 | ? | — |
| perplexity | sonar-pro | ? | — |

> **重要：** 请登录 https://app.cloubic.com 查看模型市场，确认可用模型列表后填写此表。

---

## 回退方案

如果 Cloubic 不支持某些模型，可采用混合模式：
- Cloubic 支持的模型走 Cloubic 网关
- 不支持的模型保留原有直连方式

实现方式：`config.py` 中按模型判断，支持的用 `CLOUBIC_API_KEY` + `CLOUBIC_BASE_URL`，不支持的用原始 `*_API_KEY` + 原始 `base_url`。

---

## 预期效果

| 指标 | 迁移前 | 迁移后 |
|------|--------|--------|
| API Key 数量 | 9个 | 1个 |
| 代理配置 | 需要（国外LLM） | 不需要 |
| 面板调用耗时 | 顺序（N × 单次耗时） | 并行（≈ 最慢单次耗时） |
| 代码复杂度 | 代理判断 + 多 base_url | 统一直连 |
