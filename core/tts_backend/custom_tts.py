import os
import shutil
from pathlib import Path
from gradio_client import Client, handle_file
from core.utils import load_key

def get_ref_audio(method_prefix, save_path):
    """
    根据配置获取参考音频路径
    method_prefix: 'index_tts' 或 'fish_speech'
    """
    ref_mode = load_key(f"{method_prefix}.ref_mode")
    
    # 模式 2: 使用 UI 上传的固定参考音频
    if ref_mode == 2:
        fixed_path = f"output/{method_prefix.split('_')[0]}_fixed_ref.wav"
        if os.path.exists(fixed_path):
            return fixed_path, "固定参考"
    
    # 模式 1: 使用原角色音频参考 (动态切片)
    full_stem = Path(save_path).stem
    segment_number = full_stem.split('_')[0]
    dynamic_path = os.path.join("output", "audio", "refers", f"{segment_number}.wav")
    
    if os.path.exists(dynamic_path):
        return dynamic_path, f"原片参考({segment_number}.wav)"
    
    # 兜底：如果动态切片不存在，尝试用 1.wav
    fallback_path = os.path.join("output", "audio", "refers", "1.wav")
    if os.path.exists(fallback_path):
        return fallback_path, "兜底参考(1.wav)"
        
    return None, None

def custom_tts(text, save_path):
    # 1. 获取当前选中的 TTS 方法
    method = load_key("tts_method")
    
    try:
        if method == "IndexTTS2":
            return handle_index_tts(text, save_path)
        elif method == "Fish-Speech":
            return handle_fish_speech(text, save_path)
        else:
            print(f"❌ 未知的 TTS 方法: {method}")
            return False
    except Exception as e:
        print(f"❌ {method} 运行异常: {str(e)}")
        return False

def handle_index_tts(text, save_path):
    url = load_key("index_tts.url") or "http://localhost:7860/"
    ref_path, ref_type = get_ref_audio("index_tts", save_path)
    
    if not ref_path:
        print("❌ IndexTTS2 找不到任何参考音频")
        return False

    print(f"🎙️ IndexTTS2 [{ref_type}] -> {Path(save_path).name}")
    
    client = Client(url)
    result = client.predict(
        emo_control_method="与音色参考音频相同",
        prompt=handle_file(ref_path),
        text=text,
        emo_ref_path=handle_file(ref_path),
        emo_weight=0.65,
        vec1=0, vec2=0, vec3=0, vec4=0, vec5=0, vec6=0, vec7=0, vec8=0,
        emo_text="",
        emo_random=False,
        max_text_tokens_per_segment=120,
        param_16=True, param_17=0.8, param_18=30, param_19=0.8,
        param_20=0, param_21=3, param_22=10, param_23=1500,
        api_name="/gen_single"
    )
    
    # 提取路径 (针对 IndexTTS2 的字典结构)
    real_path = result.get('value') if isinstance(result, dict) else result
    return finalize_audio(real_path, save_path)

def handle_fish_speech(text, save_path):
    url = load_key("fish_speech.url") or "http://localhost:7860/"
    ref_path, ref_type = get_ref_audio("fish_speech", save_path)
    ref_text = load_key("fish_speech.ref_text") or ""
    
    if not ref_path:
        print("❌ Fish-Speech 找不到任何参考音频")
        return False

    print(f"🐠 Fish-Speech [{ref_type}] -> {Path(save_path).name}")
    
    client = Client(url)
    result = client.predict(
        text=text,
        normalize=True,
        reference_id="videolingo_user",
        reference_audio=handle_file(ref_path),
        reference_text=ref_text,
        max_new_tokens=0,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        seed=42,
        use_memory_cache="on",
        api_name="/partial"
    )
    
    # 提取路径 (针对 Fish-Speech 的元组结构)
    real_path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(real_path, dict):
        real_path = real_path.get('name') or real_path.get('path') or real_path.get('value')
        
    return finalize_audio(real_path, save_path)

def finalize_audio(temp_path, save_path):
    """验证并移动音频文件"""
    if temp_path and os.path.exists(temp_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(temp_path, save_path)
        return True
    return False