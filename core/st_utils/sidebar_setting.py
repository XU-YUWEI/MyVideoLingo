import streamlit as st
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *
import matplotlib.font_manager as fm

@st.cache_data
def get_windows_fonts():
    # 获取系统中所有字体的名称
    fonts = fm.findSystemFonts()
    font_names = set()
    for font in fonts:
        try:
            # 提取字体的内部名称（如 Microsoft YaHei）
            font_names.add(fm.FontProperties(fname=font).get_name())
        except:
            continue
    # 过滤掉以 @ 开头的竖排字体（Windows 特有），并排序
    return sorted([f for f in font_names if not f.startswith('@')])

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

def _is_ollama_url(base_url):
    return "11434" in base_url or "ollama" in base_url.lower()

@st.cache_data(ttl=30)
def ollama_running() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

@st.cache_data(ttl=30)
def installed_qwen_models() -> list:
    import subprocess
    try:
        out = subprocess.run([r"D:\Ollama\ollama.exe", "list"], capture_output=True,
                             text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        models = []
        for line in out.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                name = parts[0].split(':')[0]
                if name.startswith("qwen3-") and name not in models:
                    models.append(name)
        return models
    except Exception:
        return []

def start_ollama_service():
    import subprocess
    try:
        subprocess.Popen([r"D:\Ollama\ollama.exe", "serve"], cwd=r"D:\Ollama",
                         creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        st.error(str(e))

def page_setting():

    display_language = st.selectbox("Display Language 🌐", 
                                  options=list(DISPLAY_LANGUAGES.keys()),
                                  index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key")
        config_input(t("BASE_URL"), "api.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))
        
        c1, c2 = st.columns([4, 1])
        with c1:
            config_input(t("MODEL"), "api.model", help=t("click to check API validity")+ " 👉")
        with c2:
            if st.button("📡", key="api"):
                st.toast(t("API Key is valid") if check_api() else t("API Key is invalid"), 
                        icon="✅" if check_api() else "❌")
        llm_support_json = st.toggle(t("LLM JSON Format Support"), value=load_key("api.llm_support_json"), help=t("Enable if your LLM supports JSON mode output"))
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

        ollama_native = st.toggle(t("Ollama Native API (Disable Thinking)"), value=load_key("api.use_ollama_native"),
                                  help=t("When using Ollama locally, call the native /api/chat endpoint with thinking disabled, speeding up qwen3 translation about 10x"))
        if ollama_native != load_key("api.use_ollama_native"):
            update_key("api.use_ollama_native", ollama_native)
            st.rerun()

        # --- 本地 Ollama：服务状态/启动按钮 + 模型切换（仅 base_url 指向 Ollama 时显示）---
        if _is_ollama_url(load_key("api.base_url")):
            st.divider()
            st.markdown(f"**{t('Ollama Service')}**")
            _running = ollama_running()
            if _running:
                st.success(t("Ollama Running"))
            else:
                st.warning(t("Ollama Not Running"))
            if st.button(t("Start Ollama Service"), key="start_ollama", disabled=_running):
                start_ollama_service()
                st.toast(t("Ollama Starting..."))
                st.rerun()
            installed = installed_qwen_models()
            if installed:
                current_model = load_key("api.model")
                index = installed.index(current_model) if current_model in installed else 0
                sel_model = st.selectbox(t("Local Model"), options=installed, index=index,
                                         help=t("Switch local model. Only installed models are listed."))
                if sel_model != current_model:
                    update_key("api.model", sel_model)
                    st.rerun()
            else:
                st.caption(t("No local qwen model installed"))
    with st.expander(t("Translation Settings"), expanded=True):
        one_shot = st.toggle(t("One-shot Translation"), value=load_key("translate_one_shot"),
                             help=t("When enabled and the whole subtitle text is short enough (translate_one_shot_max_chars), all lines are translated in a single LLM call to save tokens"))
        if one_shot != load_key("translate_one_shot"):
            update_key("translate_one_shot", one_shot)
            st.rerun()

        compact = st.toggle(t("Compact Translation Output"), value=load_key("compact_translate_output"),
                            help=t("When enabled, translation responses do not echo the original text, cutting output tokens by about one third"))
        if compact != load_key("compact_translate_output"):
            update_key("compact_translate_output", compact)
            st.rerun()

        heuristic = st.toggle(t("Heuristic-First Sentence Split"), value=load_key("split_meaning_heuristic_first"),
                              help=t("When enabled, long sentences are split locally at punctuation/conjunctions first, falling back to LLM only when the heuristic fails"))
        if heuristic != load_key("split_meaning_heuristic_first"):
            update_key("split_meaning_heuristic_first", heuristic)
            st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja"
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language"))
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtime = st.selectbox(t("WhisperX Runtime"), options=["local", "cloud", "elevenlabs"], index=["local", "cloud", "elevenlabs"].index(load_key("whisper.runtime")), help=t("Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"))
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

        if runtime == "local":
            # 说话人分离（Pyannote 本地模型，结果写入 speaker_id）
            diarize = st.checkbox("说话人分离", value=bool(load_key("whisper.diarize")), help="使用本地 Pyannote 模型进行说话人分离，结果写入 speaker_id 列")
            if diarize != load_key("whisper.diarize"):
                update_key("whisper.diarize", diarize)
                st.rerun()
            if diarize:
                num_speakers = st.number_input("期望说话人数 (0=自动)", min_value=0, max_value=50, step=1, value=int(load_key("whisper.num_speakers") or 0))
                if num_speakers != load_key("whisper.num_speakers"):
                    update_key("whisper.num_speakers", int(num_speakers))

         # --- 1. WhisperX 模型选择 ---
        model_options = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]
        current_model = load_key("whisper.model_size") or "large-v3"
        selected_model = st.selectbox("WhisperX 模型", options=model_options, index=model_options.index(current_model))
        if selected_model != current_model:
            update_key("whisper.model_size", selected_model)
            st.rerun() 
        
         # 字体选择
        # 获取系统字体列表
        all_system_fonts = get_windows_fonts()
        
        # 确定默认值（如果配置里的字体不在系统里，默认选第一个）
        current_font = load_key("subtitle.font") or "Arial"
        if current_font not in all_system_fonts:
            default_index = all_system_fonts.index("Arial") if "Arial" in all_system_fonts else 0
        else:
            default_index = all_system_fonts.index(current_font)

        # 替换原有的 selectbox
        selected_font = st.selectbox(
            "字幕字体", 
            options=all_system_fonts, 
            index=default_index
        )
        
        if selected_font != current_font:
            update_key("subtitle.font", selected_font)
            st.rerun()
            
        # --- 新增：字号选择 ---
        current_font_size = load_key("subtitle.font_size") or 17
        selected_font_size = st.number_input(
            "字体大小", 
            min_value=10, 
            max_value=100, 
            value=int(current_font_size),
            step=1
        )
        if selected_font_size != current_font_size:
            update_key("subtitle.font_size", selected_font_size)
            st.rerun()    
        
        current_color = load_key("subtitle.trans_color_hex") or "#00FFFF"
        selected_color = st.color_picker("Translation Color", value=current_color)
        if selected_color != current_color:
            update_key("subtitle.trans_color_hex", selected_color)
            ass_color = f"&H{selected_color[5:7]}{selected_color[3:5]}{selected_color[1:3]}"
            update_key("subtitle.trans_color", ass_color)
            st.rerun()

        # --- 新增：字幕背景开关 ---
        current_bg_state = load_key("subtitle.use_bg") if load_key("subtitle.use_bg") is not None else True
        use_bg = st.toggle("字幕背景 (Background)", value=current_bg_state)
        if use_bg != current_bg_state:
            update_key("subtitle.use_bg", use_bg)
            st.rerun()        
                
        with c2:
            target_language = st.text_input(t("Target Lang"), value=load_key("target_language"), help=t("Input any language in natural language, as long as llm can understand"))
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()
                        
        demucs = st.toggle(t("Vocal separation enhance"), value=load_key("demucs"), help=t("Recommended for videos with loud background noise, but will increase processing time"))
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()
        
        burn_subtitles = st.toggle(t("Burn-in Subtitles"), value=load_key("burn_subtitles"), help=t("Whether to burn subtitles into the video, will increase processing time"))
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = ["azure_tts", "openai_tts", "fish_tts", "sf_fish_tts", "edge_tts", "gpt_sovits", "custom_tts", "sf_cosyvoice2", "f5tts","IndexTTS2", "Fish-Speech"]
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(load_key("tts_method")))
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

         # --- 新增 IndexTTS2 的配置逻辑 ---
        if select_tts == "IndexTTS2":
            config_input("API URL", "index_tts.url", help="Default: http://localhost:7860/")
            
            # 参考模式选择
            index_ref_modes = {1: "使用原角色音频参考", 2: "使用固定参考音频"}
            current_mode = load_key("index_tts.ref_mode") or 1
            selected_mode = st.selectbox("参考音频方式", 
                                        options=list(index_ref_modes.keys()), 
                                        format_func=lambda x: index_ref_modes[x],
                                        index=list(index_ref_modes.keys()).index(current_mode))
            
            if selected_mode != current_mode:
                update_key("index_tts.ref_mode", selected_mode)
                st.rerun()

            # 如果选择固定参考音频，显示上传窗口
            if selected_mode == 2:
                uploaded_file = st.file_uploader("上传固定参考音频", type=["wav", "mp3"], key="index_upload")
                if uploaded_file is not None:
                    # 保存上传的文件到指定目录供后端使用
                    save_path = "output/index_fixed_ref.wav"
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"已上传: {uploaded_file.name}")

        # --- 新增 Fish-Speech 的配置逻辑 ---
        elif select_tts == "Fish-Speech":
            config_input("API URL", "fish_speech.url", help="Default: http://localhost:7860/")
            
            fish_ref_modes = {1: "使用原角色音频参考", 2: "使用固定参考音频"}
            current_mode = load_key("fish_speech.ref_mode") or 1
            selected_mode = st.selectbox("参考音频方式", 
                                        options=list(fish_ref_modes.keys()), 
                                        format_func=lambda x: fish_ref_modes[x],
                                        index=list(fish_ref_modes.keys()).index(current_mode))
            
            if selected_mode != current_mode:
                update_key("fish_speech.ref_mode", selected_mode)
                st.rerun()

            if selected_mode == 2:
                uploaded_file = st.file_uploader("上传固定参考音频", type=["wav", "mp3"], key="fish_upload")
                if uploaded_file is not None:
                    save_path = "output/fish_fixed_ref.wav"
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"已上传: {uploaded_file.name}")
                
                # Fish-Speech 建议配上参考文本
                config_input("参考音频对应的文本", "fish_speech.ref_text")
                
        # sub settings for each tts method
        elif select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")
            
            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic")
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode")) if load_key("sf_fish_tts.mode") in mode_options.keys() else 0
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(t("Fish TTS Character"), options=list(load_key("fish_tts.character_id_dict").keys()), index=list(load_key("fish_tts.character_id_dict").keys()).index(load_key("fish_tts.character")))
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")
        
        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")
            
            refer_mode_options = {1: t("Mode 1: Use provided reference audio only"), 2: t("Mode 2: Use first audio from video as reference"), 3: t("Mode 3: Use each audio from video as reference")}
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(load_key("gpt_sovits.refer_mode")),
                help=t("Configure reference audio mode for GPT-SoVITS")
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()
                
        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")
        
        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")
        
def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.", 
                      resp_type="json", log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False
    
if __name__ == "__main__":
    check_api()
