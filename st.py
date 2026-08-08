import streamlit as st
import os, sys
import debugpy
from core.st_utils.imports_and_utils import *
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 调试器初始化，确保只调用一次
# if not hasattr(sys, '_debugpy_listen_called'):
#     try:
#         debugpy.listen(5678)
#         sys._debugpy_listen_called = True
#         print("调试器已启动，等待 VS Code 附加...")
#     except RuntimeError as e:
#         if "already been called" not in str(e):
#             raise

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """, unsafe_allow_html=True)

        # ── 展示最近一次处理耗时（读取 output/log/processing_time.log 最后一条记录）──
        TIME_LOG = os.path.join("output", "log", "processing_time.log")
        if os.path.exists(TIME_LOG):
            with open(TIME_LOG, "r", encoding="utf-8") as f:
                log_lines = [ln.rstrip("\n") for ln in f.readlines()]
            last_start = None
            for i in range(len(log_lines) - 1, -1, -1):
                if log_lines[i].startswith("─────"):
                    last_start = i
                    break
            if last_start is not None:
                record = log_lines[last_start:]
                if record and record[-1] == "":
                    record = record[:-1]
                st.expander("⏱️ 最近一次处理耗时", expanded=False).markdown(
                    "\n\n".join(record))

        show_chunk = st.toggle(t("Show translation chunk progress"), value=True,
                               help=t("Show per-chunk translation progress and average speed"))
        if not os.path.exists(SUB_VIDEO):
            if st.button(t("Start Processing Subtitles"), key="text_processing_button"):
                if process_text(show_chunk):
                    st.rerun()
        else:
            if load_key("burn_subtitles"):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))
            
            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True

def process_text(show_chunk_progress=True):
    """执行字幕处理管线；显示整体进度条与每步耗时（可选翻译 chunk 级细化进度）。
    某一步失败时在页面上明确报错并返回 False（不 rerun，保留错误提示）。
    每步耗时与总耗时会追加写入 output/log/processing_time.log，历史记录不被覆盖"""
    import time

    def fmt_dur(sec):
        sec = int(round(sec))
        if sec >= 60:
            return f"{sec // 60} 分 {sec % 60} 秒"
        return f"{sec} 秒"

    # ── 时间日志：追加模式持久化，记录每次运行的历史耗时 ──
    TIME_LOG = os.path.join("output", "log", "processing_time.log")
    os.makedirs(os.path.dirname(TIME_LOG), exist_ok=True)
    video_name = ""
    try:
        from core._1_ytdlp import find_video_files
        video_name = os.path.basename(find_video_files() or "")
    except Exception:
        video_name = ""

    def _log_line(line: str):
        """向日志文件追加一行（追加模式，保留历史）"""
        try:
            with open(TIME_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[VideoLingo] 写入时间日志失败: {e}", flush=True)

    _log_line(f"───── {time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 视频: {video_name or '未知'} ─────")

    status = st.status(t("Subtitle processing in progress..."), expanded=True)
    progress = st.progress(0.0, text="")

    def _run(step_name, step_func):
        s_t0 = time.time()
        status.write(f"⏳ {step_name}")
        print(f"[VideoLingo] 步骤开始: {step_name}", flush=True)
        try:
            step_func()
        except Exception as e:
            el = time.time() - s_t0
            print(f"[VideoLingo] 步骤失败: {step_name} -> {e}", flush=True)
            status.write(f"❌ {step_name} — 失败（用时 {fmt_dur(el)}）")
            _log_line(f"  ❌ {step_name}: 失败（用时 {fmt_dur(el)}）")
            st.error(f"❌ 步骤「{step_name}」执行失败，错误信息：\n{e}\n\n修复后可直接再次点击「Start Processing Subtitles」，已完成的步骤会自动跳过。")
            st.exception(e)
            return False
        el = time.time() - s_t0
        print(f"[VideoLingo] 步骤完成: {step_name}", flush=True)
        _log_line(f"  {step_name}: {fmt_dur(el)}")
        return el

    def _summarize_and_translate():
        _4_1_summarize.get_summary()
        if load_key("pause_before_translate"):
            input(t("⚠️ PAUSE_BEFORE_TRANSLATE. Go to `output/log/terminology.json` to edit terminology. Then press ENTER to continue..."))
        if show_chunk_progress:
            # 翻译步骤映射到整体进度条 40%~60% 区间，并显示 chunk 数/平均耗时
            _t0 = time.time()
            def _cb(done, total):
                frac = 0.4 + 0.2 * (done / total) if total else 0.6
                avg = fmt_dur((time.time() - _t0) / done) if done else "-"
                progress.progress(min(frac, 0.6), text=f"{done}/{total} chunk，平均 {avg}/chunk")
            _4_2_translate.translate_all(progress_callback=_cb)
        else:
            _4_2_translate.translate_all()

    steps = [
        (t("Using Whisper for transcription..."), lambda: _2_asr.transcribe()),
        (t("Splitting long sentences..."), lambda: (_3_1_split_nlp.split_by_spacy(), _3_2_split_meaning.split_sentences_by_meaning())),
        (t("Summarizing and translating..."), _summarize_and_translate),
        (t("Processing and aligning subtitles..."), lambda: (_5_split_sub.split_for_sub_main(), _6_gen_sub.align_timestamp_main())),
        (t("Merging subtitles to video..."), _7_sub_into_vid.merge_subtitles_to_video),
    ]

    total_t0 = time.time()
    done_steps = 0
    for step_name, step_func in steps:
        el = _run(step_name, step_func)
        if el is False:
            return False
        done_steps += 1
        progress.progress(done_steps / len(steps), text="")
        status.write(f"✅ {step_name} — 用时 {fmt_dur(el)}")

    total_el = fmt_dur(time.time() - total_t0)
    _log_line(f"── 总耗时: {total_el} ──")
    _log_line("")
    progress.progress(1.0, text="")
    status.update(label=f"🎉 {t('Subtitle processing complete!')} 总耗时 {total_el}", state="complete")
    st.success(t("Subtitle processing complete! 🎉"))
    st.balloons()
    return True

def audio_processing_section():
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """, unsafe_allow_html=True)
        if not os.path.exists(DUB_VIDEO):
            if st.button(t("Start Audio Processing"), key="audio_processing_button"):
                process_audio()
                st.rerun()
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO) 
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()

def process_audio():
    with st.spinner(t("Generate audio tasks")): 
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
    with st.spinner(t("Extract refer audio")):
        _9_refer_audio.extract_refer_audio_main()
    with st.spinner(t("Generate all audio")):
        _10_gen_audio.gen_audio()
    with st.spinner(t("Merge full audio")):
        _11_merge_audio.merge_full_audio()
    with st.spinner(t("Merge dubbing to the video")):
        _12_dub_to_vid.merge_video_audio()
    
    st.success(t("Audio processing complete! 🎇"))
    st.balloons()

def main():
    logo_col, _ = st.columns([1,1])
    with logo_col:
        st.image("docs/logo.png", use_column_width=True)
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t("Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>! You can also try out our SaaS website at <a href=\"https://videolingo.io\" target=\"_blank\">videolingo.io</a> for free!")
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    download_video_section()
    text_processing_section()
    audio_processing_section()

if __name__ == "__main__":
    main()
