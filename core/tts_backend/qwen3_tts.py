"""Qwen3TTS 接入（调用 Qwen3TTS_Pro WebUI，gradio_client /fn_voice_clone）

音色来源：
  1 原角色动态参考（output/audio/refers/{段号}.wav，逐段切片）
  2 固定参考音频（侧边栏上传，output/qwen3_fixed_ref.wav）
  3 固化音色（读 personas 目录下 {名称}.wav + {名称}.txt）
另外逐行配音页的 ref_choice ('1'/'2') 对应 output/audio/user_ref{1,2}.wav
"""
import os
import shutil
from pathlib import Path

from gradio_client import Client, handle_file

from core.tts_backend.custom_tts import get_ref_audio, USER_REF_DIR
from core.utils import load_key, except_handler

DEFAULT_URL = "https://127.0.0.1:7869/"

# 复用的 gradio_client 实例（按 url 缓存，避免每句重连）
_client_cache = {}

# 服务端 fn_voice_clone 是否支持 instruct 参数（一次性探测）
_server_supports_instruct = None


def list_personas(persona_dir=None):
    """扫描固化音色目录，返回音色名称列表（ref_mode=3 时侧边栏下拉用）"""
    persona_dir = persona_dir or load_key("qwen3_tts.persona_dir") or ""
    if not persona_dir or not os.path.isdir(persona_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(persona_dir) if f.endswith(".wav"))


def _get_client(url):
    """获取（或创建）对应 url 的 gradio_client 实例"""
    if url not in _client_cache:
        _client_cache[url] = Client(url, ssl_verify=False, verbose=False)
    return _client_cache[url]


def _fn_supports_instruct(client):
    """探测服务端 fn_voice_clone 是否接受 instruct 参数"""
    global _server_supports_instruct
    if _server_supports_instruct is not None:
        return _server_supports_instruct
    try:
        api = client.view_api(return_format="dict")
        endpoints = api.get("named_endpoints", {}) or {}
        for name, info in endpoints.items():
            if name.endswith("/fn_voice_clone"):
                params = info.get("parameters", [])
                _server_supports_instruct = any(
                    p.get("parameter_name") == "instruct" for p in params
                )
                break
        else:
            _server_supports_instruct = False
    except Exception:
        _server_supports_instruct = False
    if not _server_supports_instruct:
        print("⚠️ 当前 Qwen3TTS 服务端不支持情感指令（fn_voice_clone 无 instruct 参数），已忽略")
    return _server_supports_instruct


def _resolve_ref(save_path, ref_choice=None, ref_file=None, persona_name=None):
    """解析参考音频，返回 (ref_audio_path, ref_text, 来源描述)

    ref_file:    模式2 下按行选定的固定音频文件名（refs_dir 内），空则回落全局固定参考
    persona_name: 模式3 下按行选定的固化音色名（personas 目录内），空则回落全局 persona_name
    """
    ref_mode = int(load_key("qwen3_tts.ref_mode") or 1)

    # 模式 3: 固化音色（读 personas 目录）
    if ref_mode == 3:
        persona_dir = load_key("qwen3_tts.persona_dir") or ""
        # 按行指定的音色不存在时，回退到全局 persona_name
        candidates = [persona_name or "", load_key("qwen3_tts.persona_name") or ""]
        used = None
        for name in candidates:
            if not name:
                continue
            wav_path = os.path.join(persona_dir, f"{name}.wav")
            if os.path.exists(wav_path):
                used = (name, wav_path)
                break
            if name == persona_name:
                print(f"⚠️ Qwen3TTS: 固化音色 {name} 不存在，尝试回退默认音色")
        if used is None:
            print("❌ Qwen3TTS: 未找到可用固化音色，请在页面或侧边栏选择")
            return None, None, None
        name, wav_path = used
        txt_path = os.path.join(persona_dir, f"{name}.txt")
        ref_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        return wav_path, ref_text, f"固化音色({name})"

    # 模式 2: 固定音频参考（每行可从音频库选文件）
    if ref_mode == 2 and ref_file:
        refs_dir = load_key("qwen3_tts.refs_dir") or "output/qwen_refs"
        wav_path = os.path.join(refs_dir, ref_file)
        if os.path.exists(wav_path):
            return wav_path, "", f"固定音频({ref_file})"
        print(f"⚠️ Qwen3TTS: 固定音频 {ref_file} 不存在，尝试回退默认固定参考")

    # 模式 1/2 默认及用户上传参考：复用 custom_tts 的参考音频解析
    ref_path, ref_type = get_ref_audio("qwen3_tts", save_path, ref_choice=ref_choice)
    return ref_path, "", ref_type


def _extract_audio_path(result):
    """从 gradio_client 返回值中提取音频文件路径（兼容 str / dict / tuple）"""
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, dict):
        return result.get("path") or result.get("value") or result.get("name")
    if isinstance(result, str) and result:
        return result
    return None


@except_handler("Failed to generate audio using Qwen3TTS", retry=3, delay=1)
def qwen3_tts_for_videolingo(text, save_path, number, task_df, ref_choice=None, instruct=None, ref_file=None, persona_name=None):
    """
    ref_choice: 可选 '1'/'2'，指定使用哪个用户上传的参考音频（仅模式1）
    instruct: 情感指令（按行传入），为空时回落到 config 的 qwen3_tts.instruct
    ref_file: 模式2 下按行选定的固定音频文件名（refs_dir 内）
    persona_name: 模式3 下按行选定的固化音色名（personas 目录内）
    """
    url = load_key("qwen3_tts.url") or DEFAULT_URL
    size = load_key("qwen3_tts.size") or "0.6B"
    if size not in ("1.7B", "0.6B"):
        size = "0.6B"

    ref_audio, ref_text, ref_desc = _resolve_ref(save_path, ref_choice, ref_file, persona_name)
    if not ref_audio:
        raise Exception("Qwen3TTS: 没有可用的参考音频")

    print(f"🎙️ Qwen3TTS [{ref_desc}] -> {Path(save_path).name}")

    client = _get_client(url)

    # 情感指令：只要服务端声明了 instruct 参数就必须传入（可能为空字符串，服务端会忽略），
    # 否则 gradio_client 会因参数缺失报 "No value provided for required argument"
    eff_instruct = instruct or load_key("qwen3_tts.instruct") or ""

    args = [text, "Auto", handle_file(ref_audio), ref_text or "", size]
    if _fn_supports_instruct(client):
        args.append(eff_instruct)
    result = client.predict(*args, api_name="/fn_voice_clone")

    audio_path = _extract_audio_path(result)
    if not audio_path or not os.path.exists(audio_path):
        status = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result
        raise Exception(f"Qwen3TTS 生成失败: {status}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    shutil.copy(audio_path, save_path)
    return True


if __name__ == "__main__":
    print("personas:", list_personas())
    qwen3_tts_for_videolingo("你好，这是 Qwen3TTS 测试。", "output/audio/tmp/qwen3_test.wav", 1, None)
