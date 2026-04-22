import os
import shutil
from pathlib import Path
from gradio_client import Client, handle_file

# --- 配置区 ---
INDEX_TTS_URL = "http://localhost:7860/"
# --------------

def custom_tts(text, save_path):
    try:
        # 1. 提取段落编号
        full_stem = Path(save_path).stem
        segment_number = full_stem.split('_')[0] 
        
        # 2. 定位参考音频
        ref_audio_path = os.path.join("output", "audio", "refers", f"{segment_number}.wav")
        if not os.path.exists(ref_audio_path):
            ref_audio_path = os.path.join("output", "audio", "refers", "1.wav")
            
        if not os.path.exists(ref_audio_path):
            print(f"❌ 警告：未找到参考音频 {ref_audio_path}")
            return False

        print(f"🎙️ IndexTTS2 正在处理: {full_stem} (文字: {text[:10]}...)")

        # 3. 调用 API
        client = Client(INDEX_TTS_URL)
        result = client.predict(
            emo_control_method="与音色参考音频相同",
            prompt=handle_file(ref_audio_path),
            text=text,
            emo_ref_path=handle_file(ref_audio_path),
            emo_weight=0.65,
            vec1=0, vec2=0, vec3=0, vec4=0, vec5=0, vec6=0, vec7=0, vec8=0,
            emo_text="",
            emo_random=False,
            max_text_tokens_per_segment=120,
            param_16=True, param_17=0.8, param_18=30, param_19=0.8,
            param_20=0, param_21=3, param_22=10, param_23=1500,
            api_name="/gen_single"
        )
        
        # --- 核心修复：精准抓取 'value' 键 ---
        real_temp_path = None
        if isinstance(result, dict):
            # 针对你日志中的结构：优先找 'value'，找不到再找 'name' 或 'path'
            real_temp_path = result.get('value') or result.get('name') or result.get('path')
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            # 如果返回的是列表，处理第一个元素
            first_res = result[0]
            if isinstance(first_res, dict):
                real_temp_path = first_res.get('value') or first_res.get('name')
            else:
                real_temp_path = first_res
        else:
            real_temp_path = result

        # 4. 检查并复制文件
        if real_temp_path and isinstance(real_temp_path, str) and os.path.exists(real_temp_path):
            # 确保目标文件夹存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy(real_temp_path, save_path)
            print(f"✅ 配音成功保存: {save_path}")
            return True
        else:
            print(f"❌ 错误：无法从返回结果中解析有效路径。收到结果: {result}")
            return False

    except Exception as e:
        print(f"❌ custom_tts 异常: {str(e)}")
        return False