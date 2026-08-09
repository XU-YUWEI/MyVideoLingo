import os
import json
import re
import urllib.request
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title, model):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type and item.get("model") == model:
                        return item["resp"]
        return False

def _extract_json_text(resp_content: str):
    """从模型响应中可靠提取 JSON 文本。

    本地 qwen3 模型即使关闭思考模式，也可能在 JSON 前后输出大段分析文本
    （甚至残留 <think>...</think> 块），直接交给 json_repair 会把整段混合
    文本解析成错误结构。此处按优先级提取：
    1) 剥离 <think>...</think> 块
    2) 优先取 Markdown ```json / ``` 代码块
    3) 否则从最右侧开括号开始做括号配对，截取完整闭合的 JSON 对象

    若找不到闭合的 JSON（输出被 max_tokens 截断、或响应里根本没有 JSON），
    返回 None，由调用方抛错触发重试，避免把分析文本里的碎片数组误当结果。
    """
    text = str(resp_content)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    blocks = re.findall(r'```(?:json)?\s*(.*?)```', text, flags=re.DOTALL)
    for candidate in reversed(blocks):
        candidate = candidate.strip()
        if candidate.startswith(('{', '[')):
            return candidate
    # 统一括号扫描：同时配对 {} 与 []，返回最后一个完整闭合的顶层 JSON 值。
    # 正确处理嵌套对象/数组（避免 rfind 内层括号导致只提取到碎片），
    # 找不到完整闭合（输出被截断）时返回 None 交由调用方触发重试。
    stack = []  # (open_char, position)
    closes = {'}': '{', ']': '['}
    last_complete = None
    for i, c in enumerate(text):
        if c in ('{', '['):
            stack.append((c, i))
        elif c in ('}', ']'):
            if stack and stack[-1][0] == closes[c]:
                _, pos = stack.pop()
                if not stack:
                    last_complete = text[pos:i + 1]
            else:
                stack.clear()  # 括号不匹配，重置扫描状态
    return last_complete


# ------------
# ask gpt once
# ------------

def _is_ollama(base_url):
    return "11434" in base_url or "ollama" in base_url.lower()

def _ask_ollama_native(base_url, model, messages, max_tokens=None, timeout=1800, format_json=False):
    """Call the Ollama native /api/chat endpoint with thinking disabled (think=false).

    The /v1 OpenAI-compatible endpoint of Ollama cannot turn off qwen3 thinking mode,
    which generates a long reasoning block before every answer (~10x slower). The native
    endpoint honors the top-level "think": false field. Timeout is generous (30 min) because
    local models generate one-shot translations of thousands of tokens at single-digit t/s.

    format_json=True 时设置 "format": "json"，Ollama 用语法约束强制输出合法 JSON。
    实测 qwen3-4k 对大 chunk 常先输出数千 token 的分析长文/编号列表而非 JSON，
    format=json 后 100% 输出 JSON 且 token 大幅下降。
    """
    root = base_url.strip('/')
    if root.endswith('/v1'):
        root = root[:-3]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if format_json:
        payload["format"] = "json"
    if max_tokens:
        payload["options"] = {"num_predict": int(max_tokens)}
    req = urllib.request.Request(
        root + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default", model=None):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    model = model or load_key("api.model")
    # check cache (per-model: switching models must not reuse another model's cached response)
    cached = _load_cache(prompt, resp_type, log_title, model)
    if cached:
        rprint("use cache response")
        return cached
    base_url = load_key("api.base_url")

    messages = [{"role": "user", "content": prompt}]

    # 本地 Ollama：走原生 /api/chat 并关闭思考模式（qwen3 思考在 /v1 接口上无法关闭，翻译慢约 10 倍）
    try:
        use_ollama_native = load_key("api.use_ollama_native")
    except KeyError:
        use_ollama_native = False
    if use_ollama_native and _is_ollama(base_url):
        try:
            max_tokens = load_key("api.max_tokens")
        except KeyError:
            max_tokens = None
        resp_content = _ask_ollama_native(base_url, model, messages, max_tokens=max_tokens, format_json=(resp_type == "json"))
    else:
        if 'ark' in base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
        elif 'v1' not in base_url:
            base_url = base_url.strip('/') + '/v1'
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
        response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

        params = dict(
            model=model,
            messages=messages,
            response_format=response_format,
            timeout=300
        )
        # 可选：配置 api.max_tokens 防止整片一次翻译/大分块时输出被截断
        try:
            max_tokens = load_key("api.max_tokens")
            if max_tokens:
                params['max_tokens'] = int(max_tokens)
        except KeyError:
            pass
        resp_raw = client.chat.completions.create(**params)
        resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        # 本地模型可能在 JSON 前后输出分析文本/思考块，先提取再解析，避免结构错乱
        extracted = _extract_json_text(resp_content)
        if extracted is None:
            # 注意：多数情况并非 max_tokens 截断，而是模型没按指令输出 JSON
            # （如 qwen3-4k 对大 chunk 输出分析长文/编号列表）。已通过 format=json 约束，
            # 若仍走到这里，多半是模型输出过短（空/错误格式）。
            raise ValueError("❎ Model output is not valid JSON (no complete JSON found in response). Check raw response in logs")
        resp = json_repair.loads(extracted)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
