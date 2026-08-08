"""
Qwen3TTS 逐行配音页面 - VideoLingo Streamlit 多页应用

功能：
1. 三种配音方式：原角色音频参考 / 固定音频参考 / 固定音色（页面级切换）
2. 每个片段可单独配音（逐行 TTS 生成），每行可选音频/音色及情感指令
3. 固定音频库批量上传、固化音色库管理（展示/新增/批量分配）
4. 保留一次性全部配音功能、已生成音频的播放预览
"""
import os
import sys
import time
import base64
import shutil
import re
import concurrent.futures
from io import BytesIO
from typing import Optional

import streamlit as st
import pandas as pd
from pydub import AudioSegment

# 确保项目根目录在 sys.path 中
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.utils import *
from core.utils.models import *
from core.utils.config_utils import load_key, update_key
from core.tts_backend.tts_main import tts_main
from core.tts_backend.qwen3_tts import list_personas
from core.asr_backend.audio_preprocess import get_audio_duration

# ── 页面配置 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen3TTS 逐行配音 - VideoLingo",
    page_icon="🎤",
    layout="wide",
)

# ── 常量 ─────────────────────────────────────────────────────
TASKS_FILE = _8_1_AUDIO_TASK           # "output/audio/tts_tasks.xlsx"
TEMP_DIR = _AUDIO_TMP_DIR                # "output/audio/tmp"
SEGS_DIR = _AUDIO_SEGS_DIR               # "output/audio/segs"

# ── SRT 文件路径 ──
SRC_SRT = os.path.join("output", "src.srt")
TRANS_SRT = os.path.join("output", "trans.srt")

# Qwen3TTS 固定音频参考的默认提示词（新增固化音色未填参考文本时使用）
QWEN_DEFAULT_REF_TEXT = "我们不对模型的任何滥用负责，请在使用之前考虑您当地的法律法规"


def get_qwen_ref_mode() -> int:
    """当前配音方式: 1 原角色音频参考, 2 固定音频参考, 3 固定音色"""
    mode = int(load_key("qwen3_tts.ref_mode") or 1)
    return mode if mode in (1, 2, 3) else 1


def get_refs_dir() -> str:
    """固定音频参考的音频库目录"""
    return load_key("qwen3_tts.refs_dir") or "output/qwen_refs"


def get_persona_dir() -> str:
    """固化音色目录"""
    return load_key("qwen3_tts.persona_dir") or ""


def list_refs_dir() -> list:
    """扫描固定音频库目录，返回音频文件名列表"""
    refs_dir = get_refs_dir()
    if not refs_dir or not os.path.isdir(refs_dir):
        return []
    return sorted(f for f in os.listdir(refs_dir)
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')))


def _save_as_wav(uploaded_file, target_path: str) -> bool:
    """把上传文件转存为 wav（优先 pydub，失败回退 ffmpeg）"""
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    try:
        audio = AudioSegment.from_file(uploaded_file)
        audio.export(target_path, format="wav")
        return True
    except Exception:
        try:
            import subprocess as sp
            temp_input = os.path.join(os.path.dirname(target_path), "_temp_input.bin")
            with open(temp_input, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            sp.run(["ffmpeg", "-y", "-i", temp_input, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "1", target_path],
                   capture_output=True, check=True)
            if os.path.exists(temp_input):
                os.remove(temp_input)
            return True
        except Exception as e:
            st.error(f"❌ 无法处理上传文件 {uploaded_file.name}: {e}")
            return False


def save_refs_library(uploaded_file) -> Optional[str]:
    """上传音频到固定音频库目录，返回文件名（失败返回 None）"""
    base = os.path.splitext(uploaded_file.name)[0]
    safe = re.sub(r'[^\w\u4e00-\u9fff-]', '_', base).strip('_') or f"ref{int(time.time())}"
    target = os.path.join(get_refs_dir(), f"{safe}.wav")
    if _save_as_wav(uploaded_file, target):
        return os.path.basename(target)
    return None


def add_new_persona(audio_file, name: str, ref_text: str) -> bool:
    """新增固化音色：写 personas/{名称}.wav 与 .txt"""
    persona_dir = get_persona_dir()
    if not persona_dir or not os.path.isdir(persona_dir):
        st.error("❌ personas 目录无效，请先在侧边栏配置 qwen3_tts.persona_dir")
        return False
    safe = re.sub(r'[^\w\u4e00-\u9fff-]', '_', name).strip('_')
    if not safe:
        st.error("❌ 音色名称不能为空")
        return False
    wav_path = os.path.join(persona_dir, f"{safe}.wav")
    txt_path = os.path.join(persona_dir, f"{safe}.txt")
    if not _save_as_wav(audio_file, wav_path):
        return False
    ref_text = ref_text.strip() or QWEN_DEFAULT_REF_TEXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(ref_text)
    return True


# ── 辅助函数 ─────────────────────────────────────────────────

def _srt_time_to_seconds(t: str) -> float:
    """
    将 SRT 时间格式 (HH:MM:SS,mmm 或 HH:MM:SS.mmm) 转换为秒。
    SRT 标准格式使用逗号分隔毫秒，但 Excel 中可能用点号。
    """
    t = t.strip()
    # 统一将逗号替换为点号，方便解析
    t = t.replace(',', '.')
    parts = t.split(':')
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split('.')
    s = int(s_parts[0])
    ms = int(s_parts[1]) if len(s_parts) > 1 else 0
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt_entries(srt_path: str) -> list:
    """
    解析 SRT 文件，返回列表，每个元素为 (start_sec, end_sec, role, text) 的元组。
    role: 行首 [角色N] 标记（如 '角色1'），无则为 None
    """
    if not os.path.exists(srt_path):
        return []
    entries = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        try:
            time_part = lines[1]
            start_str, end_str = time_part.split(' --> ')
            start_sec = _srt_time_to_seconds(start_str)
            end_sec = _srt_time_to_seconds(end_str)
            text = ' '.join(lines[2:])
            # 拆出行首 [角色N] 标记；正文保持干净，避免作为默认译文进入 TTS
            role = None
            m = re.match(r'^\[([^\]]*)\]\s*', text)
            if m:
                role = m.group(1).strip()
                text = text[m.end():]
            entries.append((start_sec, end_sec, role, text))
        except Exception:
            continue
    return entries


@st.cache_data(ttl=120)
def _cached_load_src_srt() -> list:
    """缓存加载原文 SRT 文件"""
    return parse_srt_entries(SRC_SRT)


@st.cache_data(ttl=120)
def _cached_load_trans_srt() -> list:
    """缓存加载译文 SRT 文件"""
    return parse_srt_entries(TRANS_SRT)


def get_srt_texts_by_segment(srt_entries: list, seg_start_str: str, seg_end_str: str) -> list:
    """
    从 SRT 条目中筛选出结束时间在 (seg_start, seg_end] 区间内的文本列表（左开右闭）。
    左开右闭保证相邻片段的边界 SRT 条目不会被重复匹配：
      片段 A [t0, t1] 匹配 (t0, t1] -> 边界值 t1 属于片段 A
      片段 B [t1, t2] 匹配 (t1, t2] -> 边界值 t1 不属于片段 B
    seg_start_str / seg_end_str: 来自 Excel 的 SRT 格式时间字符串 (HH:MM:SS.mmm)
    返回 [(text, role), ...]，role 为行首角色标记（无则为 None）
    """
    if not srt_entries or not seg_start_str or not seg_end_str:
        return []
    try:
        seg_start = _srt_time_to_seconds(seg_start_str)
        seg_end = _srt_time_to_seconds(seg_end_str)
    except Exception:
        return []
    matched = []
    for start_sec, end_sec, role, text in srt_entries:
        # 左开右闭：seg_start < end_sec <= seg_end
        if seg_start < end_sec <= seg_end:
            matched.append((text, role))
    return matched


def _role_sort_key(role: str) -> tuple:
    """角色排序键：按编号数字排序（角色10 排在 角色2 之后）"""
    m = re.search(r'\d+', role)
    return (int(m.group()), role) if m else (0, role)


def _norm_srt_text(text: str) -> str:
    """规范化文本用于角色匹配：去 [角色N] 前缀、标点、空格、括号等，转小写"""
    text = re.sub(r'^\[[^\]]*\]\s*', '', text or '')
    text = re.sub(r'[，。！？,.!?；;：:\s()（）\-]', '', text)
    return text.lower()


def build_line_role_map(df, trans_srt_entries):
    """
    为 Excel 每个配音行绑定角色，返回 (role_map, role_by_key, unresolved)。
    - 文本唯一：直接取该角色
    - 文本重复（跨角色）：取该行在片段内第 j 行的时间匹配角色；仍歧义则取全局顺序位置角色
    - 仍无法确定：计入 unresolved（应用时跳过）
    role_map:    dict 角色 -> [(number, line_idx), ...]，line_idx 为 Excel 行号
    role_by_key: dict (number, line_idx) -> 角色（供展示徽章，与批量应用一致）
    unresolved:  list [(number, line_idx)] 无法确定角色的行
    """
    # trans.srt: 规范化文本 -> 出现的角色集合
    text_roles = {}
    for _, _, role, txt in trans_srt_entries:
        if not role:
            continue
        nt = _norm_srt_text(txt)
        text_roles.setdefault(nt, set()).add(role)

    role_map = {}
    role_by_key = {}
    unresolved = []
    cursor = 0  # 全局顺序指针（用于歧义文本兜底）
    for idx, row in df.iterrows():
        number = int(row['number'])
        matched = get_srt_texts_by_segment(
            trans_srt_entries, row.get('start_time', ''), row.get('end_time', ''))
        ls = parse_lines(row.get('lines', ''))
        for j, line_text in enumerate(ls):
            nt = _norm_srt_text(line_text)
            roles = text_roles.get(nt, set())
            seq_role = trans_srt_entries[cursor][2] if cursor < len(trans_srt_entries) else None
            cursor += 1
            if len(roles) == 1:
                role = next(iter(roles))
            elif len(roles) > 1:
                # 段内第 j 行时间匹配角色消歧
                role = matched[j][1] if j < len(matched) and matched[j][1] else seq_role
            else:
                role = None
            if role:
                role_map.setdefault(role, []).append((number, j))
                role_by_key[(number, j)] = role
            else:
                unresolved.append((number, j))
        # 补齐：Excel lines 行数 < SRT 段内行数时（如 lines 缺行/多行被合并成一行），
        # 为剩余行直接用 SRT 时间匹配行的角色绑定，保证徽章完整显示
        for j in range(len(ls), len(matched)):
            role = matched[j][1]
            cursor += 1
            if role and (number, j) not in role_by_key:
                role_map.setdefault(role, []).append((number, j))
                role_by_key[(number, j)] = role
            elif not role:
                unresolved.append((number, j))
    return role_map, role_by_key, unresolved

@st.cache_data(ttl=60)
def _cached_load_tasks() -> pd.DataFrame:
    """缓存读取 Excel（仅在文件存在时调用）"""
    df = pd.read_excel(TASKS_FILE)
    if 'lines' not in df.columns:
        df['lines'] = None
    if 'src_lines' not in df.columns:
        df['src_lines'] = None

    # ── ref_lines: 每行对应的参考音频选择列表 ──
    if 'ref_lines' not in df.columns:
        # 向后兼容：从旧的 ref_choice 列初始化
        old_rc = '1'
        if 'ref_choice' in df.columns:
            old_rc = df['ref_choice'].iloc[0] if len(df) > 0 else '1'
        if old_rc not in ('1', '2'):
            old_rc = '1'
        # 为每一行生成与 lines 长度匹配的列表
        def _make_default_refs(raw_lines):
            ls = parse_lines(raw_lines)
            return str([old_rc] * len(ls)) if ls else str([])
        df['ref_lines'] = df['lines'].apply(_make_default_refs)
    else:
        # 确保 ref_lines 长度与 lines 匹配（修复旧数据）
        for idx, row in df.iterrows():
            refs = parse_lines(row['ref_lines'])
            ls = parse_lines(row['lines'])
            if ls and len(refs) != len(ls):
                fallback = '1'
                if 'ref_choice' in df.columns:
                    fb = str(row['ref_choice'])
                    if fb in ('1', '2'):
                        fallback = fb
                df.at[idx, 'ref_lines'] = str([fallback] * len(ls))

    # ── instruct_lines: 每行对应的情感指令列表（可选，仅 Qwen3TTS 生效）──
    if 'instruct_lines' not in df.columns:
        def _make_default_instructs(raw_lines):
            ls = parse_lines(raw_lines)
            return str([''] * len(ls)) if ls else str([])
        df['instruct_lines'] = df['lines'].apply(_make_default_instructs)
    else:
        # 确保 instruct_lines 长度与 lines 匹配（修复旧数据）
        for idx, row in df.iterrows():
            insts = parse_lines(row['instruct_lines'])
            ls = parse_lines(row['lines'])
            if ls and len(insts) != len(ls):
                df.at[idx, 'instruct_lines'] = str([''] * len(ls))

    # ── ref_files: 每行对应的音频文件名（模式2）或固化音色名（模式3），空串=默认 ──
    if 'ref_files' not in df.columns:
        def _make_default_ref_files(raw_lines):
            ls = parse_lines(raw_lines)
            return str([''] * len(ls)) if ls else str([])
        df['ref_files'] = df['lines'].apply(_make_default_ref_files)
    else:
        # 确保 ref_files 长度与 lines 匹配（修复旧数据）
        for idx, row in df.iterrows():
            rf = parse_lines(row['ref_files'])
            ls = parse_lines(row['lines'])
            if ls and len(rf) != len(ls):
                df.at[idx, 'ref_files'] = str([''] * len(ls))

    return df


def load_tasks() -> Optional[pd.DataFrame]:
    """加载配音任务 DataFrame（带文件存在性检查，不受缓存干扰）"""
    if not os.path.exists(TASKS_FILE):
        return None
    return _cached_load_tasks()


def parse_lines(raw):
    """安全地将 Excel 中存储的 lines 解析为列表"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return eval(raw)
        except Exception:
            return [raw]
    return []


def _invalidate_audio_cache(number: int = None):
    """清除指定片段（或全部片段）的音频状态缓存和时长缓存，供 dub/delete 操作后调用"""
    if number is not None:
        st.session_state.pop(f"_audio_status_{number}", None)
        st.session_state.pop(f"_audio_durations_{number}", None)
        # 原音缓存（_orig_audio_*）不清除，因为原音文件不会因配音而改变
    else:
        for key in list(st.session_state.keys()):
            if key.startswith("_audio_status_") or key.startswith("_audio_durations_"):
                del st.session_state[key]


def get_segment_status(number: int, lines_list: list) -> tuple:
    """
    检查某个片段的音频生成状态（带 session_state 缓存，避免重复 I/O）。
    返回 (all_generated: bool, audio_paths: list, combined_audio: Optional[BytesIO])
    """
    # ── 缓存命中：相同 segment number + 相同行数，直接返回 ──
    cache_key = f"_audio_status_{number}"
    cached = st.session_state.get(cache_key)
    if cached is not None and cached.get('lines_len') == len(lines_list):
        return cached['all_generated'], cached['audio_paths'], cached['combined']

    os.makedirs(TEMP_DIR, exist_ok=True)
    audio_paths = []
    all_generated = True

    for line_idx in range(len(lines_list)):
        seg_path = os.path.join(SEGS_DIR, f"{number}_{line_idx}.wav")
        tmp_path = os.path.join(TEMP_DIR, f"{number}_{line_idx}_temp.wav")

        if os.path.exists(seg_path):
            audio_paths.append(seg_path)
        elif os.path.exists(tmp_path):
            audio_paths.append(tmp_path)
        else:
            all_generated = False
            audio_paths.append(None)

    # 如果所有音频都已生成，尝试合并为一个 AudioSegment 用于播放
    combined = None
    if all_generated and audio_paths:
        try:
            combined_audio = AudioSegment.empty()
            for ap in audio_paths:
                if ap and os.path.exists(ap):
                    combined_audio += AudioSegment.from_wav(ap)
            buf = BytesIO()
            combined_audio.export(buf, format="wav")
            buf.seek(0)
            combined = buf
        except Exception:
            combined = None

    # ── 写入缓存 ──
    st.session_state[cache_key] = {
        'lines_len': len(lines_list),
        'all_generated': all_generated,
        'audio_paths': audio_paths,
        'combined': combined,
    }

    return all_generated, audio_paths, combined


def dub_single_segment(number: int, lines_list: list, tasks_df: pd.DataFrame, ref_lines: list = None, instruct_lines: list = None, ref_files: list = None) -> bool:
    """为单个配音片段生成 TTS 音频（每行可使用不同音频/音色及情感指令）"""
    if ref_lines is None:
        ref_lines = ['1'] * len(lines_list)
    if instruct_lines is None:
        instruct_lines = [''] * len(lines_list)
    if ref_files is None:
        ref_files = [''] * len(lines_list)
    # 确保列表长度匹配
    while len(ref_lines) < len(lines_list):
        ref_lines.append('1')
    while len(instruct_lines) < len(lines_list):
        instruct_lines.append('')
    while len(ref_files) < len(lines_list):
        ref_files.append('')

    os.makedirs(TEMP_DIR, exist_ok=True)
    success = True
    progress_bar = st.progress(0, text=f"正在生成片段 {number} 的音频...")

    for line_idx, line in enumerate(lines_list):
        line_ref = ref_lines[line_idx] if line_idx < len(ref_lines) else '1'
        if line_ref not in ('1', '2'):
            line_ref = '1'
        line_instruct = instruct_lines[line_idx] if line_idx < len(instruct_lines) else ''
        line_ref_file = ref_files[line_idx] if line_idx < len(ref_files) else ''
        ref_label = f"参考{line_ref}" if not line_ref_file else line_ref_file
        if line_instruct:
            ref_label += f"/指令:{line_instruct}"

        temp_file = os.path.join(TEMP_DIR, f"{number}_{line_idx}_temp.wav")
        try:
            tts_main(line, temp_file, number, tasks_df, ref_choice=None, instruct=line_instruct,
                     ref_file=line_ref_file, persona_name=line_ref_file)
            progress_bar.progress(
                (line_idx + 1) / len(lines_list),
                text=f"片段 {number}: 第 {line_idx + 1}/{len(lines_list)} 行 [{ref_label}]",
            )
        except Exception as e:
            st.error(f"片段 {number} 第 {line_idx + 1} 行配音失败: {e}")
            success = False
            break

    progress_bar.empty()
    return success


def redub_single_segment(number: int, lines_list: list, tasks_df: pd.DataFrame, ref_lines: list = None, instruct_lines: list = None, ref_files: list = None) -> bool:
    """重新配音：先删除旧音频文件，再重新生成（每行可使用不同音频/音色及情感指令）"""
    # 先删除已有的音频文件
    delete_count = delete_segment_audio(number, lines_list)
    if delete_count > 0:
        st.toast(f"已删除 {delete_count} 个旧音频文件")
    # 再重新生成
    return dub_single_segment(number, lines_list, tasks_df, ref_lines=ref_lines, instruct_lines=instruct_lines, ref_files=ref_files)


def delete_segment_audio(number: int, lines_list: list) -> int:
    """删除指定片段的音频文件（含 tmp 临时文件和 segs 最终文件）"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SEGS_DIR, exist_ok=True)
    deleted = 0
    for line_idx in range(len(lines_list)):
        # 临时文件
        tmp_path = os.path.join(TEMP_DIR, f"{number}_{line_idx}_temp.wav")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            deleted += 1
        # 最终文件
        seg_path = os.path.join(SEGS_DIR, f"{number}_{line_idx}.wav")
        if os.path.exists(seg_path):
            os.remove(seg_path)
            deleted += 1
    return deleted


def _dub_segment_worker(number: int, lines_list: list, tasks_df: pd.DataFrame, ref_lines: list, instruct_lines: list = None, ref_files: list = None) -> tuple:
    """线程安全的配音 worker——不调用任何 Streamlit UI 函数，只返回结果"""
    if instruct_lines is None:
        instruct_lines = [''] * len(lines_list)
    if ref_files is None:
        ref_files = [''] * len(lines_list)
    os.makedirs(TEMP_DIR, exist_ok=True)
    for line_idx, line in enumerate(lines_list):
        line_ref = ref_lines[line_idx] if line_idx < len(ref_lines) else '1'
        if line_ref not in ('1', '2'):
            line_ref = '1'
        line_instruct = instruct_lines[line_idx] if line_idx < len(instruct_lines) else ''
        line_ref_file = ref_files[line_idx] if line_idx < len(ref_files) else ''
        temp_file = os.path.join(TEMP_DIR, f"{number}_{line_idx}_temp.wav")
        try:
            tts_main(line, temp_file, number, tasks_df, ref_choice=None, instruct=line_instruct,
                     ref_file=line_ref_file, persona_name=line_ref_file)
        except Exception as e:
            return (number, False, str(e))
    return (number, True, None)


def get_audio_player(audio_buf: BytesIO) -> str:
    """生成 HTML audio 播放器标签"""
    audio_buf.seek(0)
    audio_bytes = audio_buf.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls style="width: 100%; height: 40px;"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'


def format_time_display(seconds: float) -> str:
    """将秒数格式化为 mm:ss.xx"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def get_original_audio_segment(number: int, start_time_str: str, end_time_str: str) -> Optional[BytesIO]:
    """
    获取指定片段的原音音频，返回 BytesIO 供播放器使用。
    优先使用已提取的参考音频文件 (output/audio/refers/{number}.wav)，
    若不存在则从原始音频文件中实时截取。

    结果缓存到 session_state（原音文件不会变化，只需加载一次）。
    """
    cache_key = f"_orig_audio_{number}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    ref_path = os.path.join(_AUDIO_REFERS_DIR, f"{number}.wav")
    if os.path.exists(ref_path):
        try:
            audio = AudioSegment.from_wav(ref_path)
            buf = BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            st.session_state[cache_key] = buf
            return buf
        except Exception:
            pass

    # 实时截取：从 vocal.mp3 或 raw.mp3 中截取对应时间段
    source_audio = None
    for src in [_VOCAL_AUDIO_FILE, _RAW_AUDIO_FILE]:
        if os.path.exists(src):
            source_audio = src
            break
    if source_audio is None:
        st.session_state[cache_key] = None
        return None

    try:
        start_sec = _srt_time_to_seconds(start_time_str)
        end_sec = _srt_time_to_seconds(end_time_str)
        duration_sec = end_sec - start_sec
        if duration_sec <= 0:
            st.session_state[cache_key] = None
            return None

        # 用 ffmpeg 精确截取片段到临时文件
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_seg = os.path.join(TEMP_DIR, f"orig_{number}.wav")
        import subprocess as sp
        sp.run([
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-i', source_audio,
            '-t', str(duration_sec),
            '-ar', '16000',
            '-ac', '1',
            temp_seg
        ], capture_output=True, check=True)

        audio = AudioSegment.from_wav(temp_seg)
        buf = BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        # 清理临时文件
        try:
            os.remove(temp_seg)
        except Exception:
            pass
        st.session_state[cache_key] = buf
        return buf
    except Exception:
        st.session_state[cache_key] = None
        return None


# ── 主页面 ───────────────────────────────────────────────────

def main():
    st.title("🎤 Qwen3TTS 逐行配音")
    st.markdown("---")

    # 步骤1: 检查任务文件是否存在
    df = load_tasks()
    if df is None or df.empty:
        st.warning("⚠️ 未找到配音任务文件。请先「生成音频任务」。")
        st.info(f"期待的文件位置: `{TASKS_FILE}`")

        # 提供快捷跳转
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 生成音频任务", type="primary", use_container_width=True):
                with st.spinner("正在生成音频任务..."):
                    try:
                        from core._8_1_audio_task import gen_audio_task_main
                        from core._8_2_dub_chunks import gen_dub_chunks
                        gen_audio_task_main()
                        gen_dub_chunks()
                        st.success("✅ 音频任务生成完成！")
                        _cached_load_tasks.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 生成音频任务失败: {e}")
        with col_b:
            if st.button("🏠 返回主页面", use_container_width=True):
                st.switch_page("st.py")
        return

    # ── 配音方式选择（映射 qwen3_tts.ref_mode） ──
    qwen_modes = {
        1: "原角色音频参考",
        2: "固定音频参考",
        3: "固定音色",
    }
    current_mode = get_qwen_ref_mode()
    sel_mode = st.radio(
        "🎤 配音方式",
        options=[1, 2, 3],
        format_func=lambda x: qwen_modes[x],
        index=current_mode - 1,
        horizontal=True,
    )
    if sel_mode != current_mode:
        update_key("qwen3_tts.ref_mode", sel_mode)
        st.rerun()
    qwen_mode = sel_mode

    if qwen_mode == 1:
        st.info("**原角色音频参考**：自动按片段序号取 `output/audio/refers/{段号}.wav` 作为参考音频，无需逐行选择（可填情感指令）。")

    elif qwen_mode == 2:
        # ── 固定音频库：批量上传 + 展示 ──
        with st.expander("🎵 固定音频库（批量上传，每行可下拉选择）", expanded=False):
            refs_dir = get_refs_dir()
            st.caption(f"音频库目录: `{refs_dir}`")
            uploaded_files = st.file_uploader(
                "批量上传音频（支持多选）",
                type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4', 'avi', 'mov', 'mkv'],
                accept_multiple_files=True,
                key="refs_lib_upload",
                label_visibility="collapsed",
            )
            if uploaded_files:
                ok_names = []
                fail_count = 0
                for uf in uploaded_files:
                    saved = save_refs_library(uf)
                    if saved:
                        ok_names.append(saved)
                    else:
                        fail_count += 1
                if ok_names:
                    st.success(f"✅ 已上传 {len(ok_names)} 个音频: {', '.join(ok_names)}")
                if fail_count:
                    st.error(f"❌ {fail_count} 个文件上传失败")
                st.rerun()
            refs_files = list_refs_dir()
            if refs_files:
                st.markdown("**当前音频库**：")
                for rf in refs_files:
                    rf_path = os.path.join(refs_dir, rf)
                    rf_col1, rf_col2 = st.columns([2, 3])
                    with rf_col1:
                        st.markdown(f"`{rf}`")
                    with rf_col2:
                        if os.path.exists(rf_path):
                            st.audio(rf_path)
            else:
                st.warning("音频库为空，请上传音频（默认固定参考为 `output/qwen3_fixed_ref.wav`）。")

    elif qwen_mode == 3:
        # ── 固定音色库：展示 + 新增 + 批量分配 ──
        with st.expander("🗣️ 固定音色库（personas 固化音色，每行可下拉选择）", expanded=True):
            persona_dir = get_persona_dir()
            st.caption(f"personas 目录: `{persona_dir}`")
            persona_names = list_personas(persona_dir) if persona_dir else []
            if persona_names:
                st.markdown("**现有固化音色**（可试听）:")
                for pname in persona_names:
                    p_col1, p_col2 = st.columns([2, 3])
                    with p_col1:
                        st.markdown(f"`{pname}`")
                    with p_col2:
                        p_wav = os.path.join(persona_dir, f"{pname}.wav")
                        if os.path.exists(p_wav):
                            st.audio(p_wav)
            else:
                st.warning("未找到固化音色，请检查 personas 目录或新增音色。")

            st.markdown("---")
            st.markdown("**➕ 新增固化音色**")
            new_col1, new_col2, new_col3 = st.columns([1, 1, 2])
            with new_col1:
                new_name = st.text_input("音色名称", key="new_persona_name", placeholder="如: 老王")
            with new_col2:
                new_audio = st.file_uploader("上传参考音频", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
                                             key="new_persona_audio", label_visibility="collapsed")
            with new_col3:
                new_ref_text = st.text_input("参考文本（可选，留空用默认提示词）", key="new_persona_reftext")
            if st.button("💾 保存新音色", use_container_width=True, key="add_persona_btn"):
                if new_audio is None:
                    st.error("❌ 请先上传参考音频")
                elif add_new_persona(new_audio, new_name, new_ref_text):
                    st.success(f"✅ 固化音色 `{new_name}` 已保存")
                    st.rerun()

            st.markdown("---")
            persona_names2 = list_personas(persona_dir) if persona_dir else []
            if persona_names2:
                st.markdown("**🔄 批量分配音色到全部片段**")
                assign_col1, assign_col2 = st.columns([3, 1])
                with assign_col1:
                    batch_persona = st.selectbox("选择音色", options=persona_names2, key="batch_persona_sel")
                with assign_col2:
                    if st.button("应用到全部", use_container_width=True, key="apply_batch_persona_btn"):
                        apply_count = 0
                        for idx2, row2 in df.iterrows():
                            ls2 = parse_lines(row2['lines'])
                            if ls2:
                                df.at[idx2, 'ref_files'] = str([batch_persona] * len(ls2))
                                apply_count += 1
                        df.to_excel(TASKS_FILE, index=False)
                        _cached_load_tasks.clear()
                        st.success(f"✅ 已批量设置 {apply_count} 个片段的音色为 {batch_persona}")
                        st.rerun()

            st.markdown("---")
            # ── 按角色批量分配音色（角色来自译文 SRT 的 [角色N] 标记） ──
            trans_srt_entries = _cached_load_trans_srt()
            role_map, _, unresolved = build_line_role_map(df, trans_srt_entries)
            if role_map:
                st.markdown("**🎭 按角色批量分配音色**")
                st.caption("每个角色选择一个音色后，点下方按钮统一应用到该角色的所有行；留空 = 不改该角色。")
                for role in sorted(role_map.keys(), key=_role_sort_key):
                    r_col1, r_col2 = st.columns([1, 2])
                    with r_col1:
                        st.markdown(f"**{role}**（{len(role_map[role])} 行）")
                    with r_col2:
                        st.selectbox(
                            f"{role} 音色",
                            options=[''] + persona_names2,
                            format_func=lambda x: "不改（保持现状）" if x == '' else x,
                            key=f"role_persona_{role}",
                            label_visibility="collapsed",
                        )
                if st.button("✅ 应用角色音色到对应行", use_container_width=True, key="apply_role_persona_btn"):
                    try:
                        num_to_idx = {int(r['number']): i for i, r in df.iterrows()}
                        applied = 0
                        changed = 0
                        for role, pairs in role_map.items():
                            persona = st.session_state.get(f"role_persona_{role}", '')
                            if not persona:
                                continue
                            # 行索引 j 为 Excel 行号，逐行写入 ref_files
                            for num, j in pairs:
                                if num not in num_to_idx:
                                    continue
                                ridx = num_to_idx[num]
                                ls2 = parse_lines(df.at[ridx, 'lines'])
                                if j >= len(ls2):
                                    # 兜底：lines 缺行时扩展并写回，保证角色音色应用到所有行
                                    while len(ls2) <= j:
                                        ls2.append('')
                                    df.at[ridx, 'lines'] = str(ls2)
                                    for _col in ('ref_lines', 'instruct_lines'):
                                        if _col in df.columns:
                                            _cur = parse_lines(df.at[ridx, _col])
                                            while len(_cur) < len(ls2):
                                                _cur.append('')
                                            df.at[ridx, _col] = str(_cur)
                                rf = parse_lines(df.at[ridx, 'ref_files'])
                                while len(rf) <= j:
                                    rf.append('')
                                if rf[j] != persona:
                                    rf[j] = persona
                                    changed += 1
                                df.at[ridx, 'ref_files'] = str(rf)
                                applied += 1
                        df.to_excel(TASKS_FILE, index=False)
                        _cached_load_tasks.clear()
                        msg = f"✅ 已为 {applied} 行设置角色音色（其中 {changed} 行发生变化）"
                        if unresolved:
                            msg += f"；另有 {len(unresolved)} 行无法确定角色，未处理"
                        st.success(msg)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 应用角色音色失败: {e}")
            else:
                st.caption("未在译文 SRT 中检测到角色标记，暂无法按角色分配音色。")

    st.markdown("---")

    # ── 页面顶部：统计信息 & 批量操作 ──
    total_segments = len(df)
    # 统计总行数
    all_lines = [parse_lines(row['lines']) for _, row in df.iterrows()]
    total_lines = sum(len(ls) for ls in all_lines)

    # 统计已配音的片段数
    dubbed_count = 0
    for _, row in df.iterrows():
        number = row['number']
        lines_list = parse_lines(row['lines'])
        generated, _, _ = get_segment_status(number, lines_list)
        if generated:
            dubbed_count += 1

    # 统计总时长（excel 中的 duration 字段，单位秒）
    total_duration = df['duration'].sum() if 'duration' in df.columns else 0

    # 统计信息行
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 总片段数", total_segments)
    with col2:
        st.metric("📝 总字幕行数", total_lines)
    with col3:
        st.metric("✅ 已配音", f"{dubbed_count}/{total_segments}")
    with col4:
        pct = (dubbed_count / total_segments * 100) if total_segments > 0 else 0
        st.metric("📊 完成度", f"{pct:.1f}%")
    with col5:
        st.metric("⏱ 总时长", format_time_display(total_duration))

    st.markdown("---")

    # 显示上次"处理音频"的结果（跨 rerun 保留）
    if "_audio_process_result" in st.session_state:
        processed, skipped = st.session_state.pop("_audio_process_result")
        if processed:
            st.success(f"✅ 已处理 {processed} 个片段（变速 + 时间轴对齐）")
        if skipped:
            msg = "；".join(f"片段 {n}: {reason}" for n, reason in skipped)
            st.warning(f"⚠️ 已跳过 {len(skipped)} 个未配音/音频不完整的片段：{msg}")

    # 批量操作按钮行
    action_col1, action_col2, action_col3, action_col4, action_col5, action_col6 = st.columns([1, 1, 1, 1, 1, 1])

    with action_col1:
        # 处理音频：只做变速 + 时间轴重建，不清理已有配音、不重新 TTS
        if st.button("⚙️ 处理音频", type="primary", use_container_width=True):
            with st.spinner("正在处理音频（变速 + 时间轴重建）..."):
                try:
                    from core._10_gen_audio import process_audio
                    processed, skipped = process_audio()
                    st.session_state["_audio_process_result"] = (processed, skipped)
                    _invalidate_audio_cache()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 处理音频失败: {e}")

    with action_col2:
        # 仅生成未配音的片段（增量式，逐行带各自音频/音色与情感指令）
        if st.button("➕ 补齐未配音", use_container_width=True):
            with st.spinner("正在为未配音的片段生成音频..."):
                success_count = 0
                fail_count = 0
                progress = st.progress(0, text="补齐中...")
                total_undubbed = total_segments - dubbed_count

                if total_undubbed == 0:
                    st.info("所有片段已配音完成！")
                else:
                    # 收集所有未配音片段
                    undubbed_tasks = []
                    for idx, row in df.iterrows():
                        number = row['number']
                        lines_list = parse_lines(row['lines'])
                        seg_ref_lines = parse_lines(row.get('ref_lines', ''))
                        if not seg_ref_lines or len(seg_ref_lines) != len(lines_list):
                            seg_ref_lines = ['1'] * len(lines_list) if lines_list else []
                        seg_instruct_lines = parse_lines(row.get('instruct_lines', ''))
                        if not seg_instruct_lines or len(seg_instruct_lines) != len(lines_list):
                            seg_instruct_lines = [''] * len(lines_list) if lines_list else []
                        seg_ref_files = parse_lines(row.get('ref_files', ''))
                        if not seg_ref_files or len(seg_ref_files) != len(lines_list):
                            seg_ref_files = [''] * len(lines_list) if lines_list else []
                        generated, _, _ = get_segment_status(number, lines_list)
                        if not generated and lines_list:
                            undubbed_tasks.append((number, lines_list, seg_ref_lines, seg_instruct_lines, seg_ref_files))

                    total = len(undubbed_tasks)
                    if total == 0:
                        st.info("所有片段已配音完成！")
                    else:
                        # 使用线程池并发配音（默认 3 个 worker）
                        max_workers = 3
                        progress_bar = st.progress(0, text=f"并行配音中 (最大 {max_workers} 并发)...")
                        completed = 0
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_map = {}
                            for number, lines_list, seg_ref_lines, seg_instruct_lines, seg_ref_files in undubbed_tasks:
                                fut = executor.submit(_dub_segment_worker, number, lines_list, df, seg_ref_lines, seg_instruct_lines, seg_ref_files)
                                future_map[fut] = number

                            for future in concurrent.futures.as_completed(future_map):
                                number, ok, err_msg = future.result()
                                completed += 1
                                progress_bar.progress(
                                    completed / total,
                                    text=f"并行配音中: {completed}/{total} 片段完成"
                                )
                                if ok:
                                    success_count += 1
                                else:
                                    fail_count += 1
                                    st.error(f"❌ 片段 #{number} 失败: {err_msg}")

                        progress_bar.empty()
                        if fail_count == 0:
                            st.success(f"✅ 补齐完成！成功生成 {success_count} 个片段。")
                        else:
                            st.warning(f"⚠️ 补齐完成。成功: {success_count}, 失败: {fail_count}")
                    _invalidate_audio_cache()
                    st.rerun()

    with action_col3:
        # 批量配音（所有片段，并发覆盖，逐行带各自音频/音色与情感指令）
        if st.button("🔊 批量配音", type="primary", use_container_width=True):
            with st.spinner("正在批量配音..."):
                # 先删除所有已有音频
                total_del = 0
                for _, row in df.iterrows():
                    number = row['number']
                    lines_list = parse_lines(row['lines'])
                    total_del += delete_segment_audio(number, lines_list)

                # 收集所有片段
                all_tasks = []
                for _, row in df.iterrows():
                    number = row['number']
                    lines_list = parse_lines(row['lines'])
                    seg_ref_lines = parse_lines(row.get('ref_lines', ''))
                    if not seg_ref_lines or len(seg_ref_lines) != len(lines_list):
                        seg_ref_lines = ['1'] * len(lines_list) if lines_list else []
                    seg_instruct_lines = parse_lines(row.get('instruct_lines', ''))
                    if not seg_instruct_lines or len(seg_instruct_lines) != len(lines_list):
                        seg_instruct_lines = [''] * len(lines_list) if lines_list else []
                    seg_ref_files = parse_lines(row.get('ref_files', ''))
                    if not seg_ref_files or len(seg_ref_files) != len(lines_list):
                        seg_ref_files = [''] * len(lines_list) if lines_list else []
                    if lines_list:
                        all_tasks.append((number, lines_list, seg_ref_lines, seg_instruct_lines, seg_ref_files))

                total = len(all_tasks)
                if total == 0:
                    st.info("没有需要配音的片段。")
                else:
                    success_count = 0
                    fail_count = 0
                    max_workers = 3
                    progress_bar = st.progress(0, text=f"批量并行配音中 (最大 {max_workers} 并发)...")
                    completed = 0
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map = {}
                        for number, lines_list, seg_ref_lines, seg_instruct_lines, seg_ref_files in all_tasks:
                            fut = executor.submit(_dub_segment_worker, number, lines_list, df, seg_ref_lines, seg_instruct_lines, seg_ref_files)
                            future_map[fut] = number

                        for future in concurrent.futures.as_completed(future_map):
                            number, ok, err_msg = future.result()
                            completed += 1
                            progress_bar.progress(
                                completed / total,
                                text=f"批量配音中: {completed}/{total} 片段完成"
                            )
                            if ok:
                                success_count += 1
                            else:
                                fail_count += 1
                                st.error(f"❌ 片段 #{number} 失败: {err_msg}")

                    progress_bar.empty()
                    if fail_count == 0:
                        st.success(f"✅ 批量配音完成！成功生成 {success_count} 个片段（已删除 {total_del} 个旧文件）。")
                    else:
                        st.warning(f"⚠️ 批量配音完成。成功: {success_count}, 失败: {fail_count}（已删除 {total_del} 个旧文件）。")
                _invalidate_audio_cache()
                st.rerun()

    with action_col4:
        # 重新生成配音任务（重新运行 _8_2_dub_chunks）
        if st.button("🔄 重新生成配音任务", use_container_width=True):
            with st.spinner("正在重新生成配音任务..."):
                try:
                    from core._8_2_dub_chunks import gen_dub_chunks
                    gen_dub_chunks()
                    st.success("✅ 配音任务重新生成完成！")
                    _cached_load_tasks.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 重新生成失败: {e}")

    with action_col5:
        # 生成音频任务（运行 _8_1_audio_task + _8_2_dub_chunks）
        if st.button("📋 生成音频任务", use_container_width=True):
            with st.spinner("正在生成音频任务..."):
                try:
                    from core._8_1_audio_task import gen_audio_task_main
                    from core._8_2_dub_chunks import gen_dub_chunks
                    gen_audio_task_main()
                    gen_dub_chunks()
                    st.success("✅ 音频任务生成完成！")
                    _cached_load_tasks.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 生成音频任务失败: {e}")

    with action_col6:
        # 删除全部配音（所有片段的音频文件）
        if st.button("🗑️ 删除全部配音", type="secondary", use_container_width=True):
            deleted = 0
            for _, row in df.iterrows():
                number = row['number']
                lines_list = parse_lines(row['lines'])
                deleted += delete_segment_audio(number, lines_list)
            if deleted > 0:
                st.success(f"✅ 已删除全部配音文件，共 {deleted} 个文件。")
            else:
                st.info("没有找到可删除的配音文件。")
            _invalidate_audio_cache()
            st.rerun()

    # ── 合并到视频 ──
    DUB_VIDEO = "output/output_dub.mp4"
    dub_video_exists = os.path.exists(DUB_VIDEO)

    with st.container(border=True):
        merge_col1, merge_col2 = st.columns([1, 4])
        with merge_col1:
            st.markdown("**🎬 合并到视频**")
        with merge_col2:
            status = "✅ 已生成" if dub_video_exists else "❌ 未生成"
            st.markdown(
                f"将已生成的配音音频合并到原始视频中，生成最终配音视频。"
                f"状态: **{status}**"
                " 流程：① 合并音频片段 → ② 合成到视频（含字幕烧录）。"
            )

        # ── 字幕设置（可折叠） ──
        with st.expander("📝 字幕设置", expanded=False):
            from core.utils.config_utils import load_key, update_key
            from core.st_utils.sidebar_setting import get_windows_fonts

            # 烧录字幕开关
            current_burn = load_key("burn_subtitles")
            burn_on = st.toggle("烧录字幕", value=current_burn if current_burn is not None else True)
            if burn_on != current_burn:
                update_key("burn_subtitles", burn_on)

            if burn_on:
                sub_font_col1, sub_font_col2 = st.columns(2)
                with sub_font_col1:
                    # 字体选择
                    all_fonts = get_windows_fonts()
                    current_font = load_key("subtitle.font") or "Arial"
                    if current_font not in all_fonts:
                        font_idx = all_fonts.index("Arial") if "Arial" in all_fonts else 0
                    else:
                        font_idx = all_fonts.index(current_font)
                    selected_font = st.selectbox("字幕字体", options=all_fonts, index=font_idx)
                    if selected_font != current_font:
                        update_key("subtitle.font", selected_font)

                    # 字号
                    current_size = load_key("subtitle.font_size") or 17
                    selected_size = st.number_input("字体大小", min_value=10, max_value=100, value=int(current_size), step=1)
                    if selected_size != current_size:
                        update_key("subtitle.font_size", selected_size)

                with sub_font_col2:
                    # 字体颜色
                    current_hex = load_key("subtitle.trans_color_hex") or "#00FFFF"
                    selected_color = st.color_picker("字体颜色", value=current_hex)
                    if selected_color != current_hex:
                        update_key("subtitle.trans_color_hex", selected_color)
                        ass_color = f"&H{selected_color[5:7]}{selected_color[3:5]}{selected_color[1:3]}"
                        update_key("subtitle.trans_color", ass_color)

                    # 字幕背景开关
                    current_bg = load_key("subtitle.use_bg") if load_key("subtitle.use_bg") is not None else True
                    use_bg = st.toggle("字幕背景", value=current_bg)
                    if use_bg != current_bg:
                        update_key("subtitle.use_bg", use_bg)

            # ── 配音音量调节 ──
            with st.container():
                st.markdown("**🔊 音频混合设置**")
                try:
                    current_dub_vol = load_key("dub_volume")
                except KeyError:
                    current_dub_vol = 0.5
                dub_vol = st.slider(
                    "配音音量（相对背景音）",
                    min_value=0.0, max_value=1.0,
                    value=float(current_dub_vol), step=0.05,
                    help="降低配音音量可让视频背景音更清晰。默认 0.5（50%）"
                )
                if dub_vol != float(current_dub_vol):
                    try:
                        update_key("dub_volume", dub_vol)
                    except KeyError:
                        pass

        if st.button("🎬 将最终音频合并到视频中", type="secondary", use_container_width=True, key="merge_to_video"):
            try:
                # Step 1: 合并音频片段
                with st.spinner("步骤 1/2: 正在合并音频片段..."):
                    from core._11_merge_audio import merge_full_audio
                    merge_full_audio()
                    st.success("✅ 音频合并完成！")

                # Step 2: 合并到视频
                with st.spinner("步骤 2/2: 正在合成到视频（含字幕烧录）..."):
                    from core._12_dub_to_vid import merge_video_audio
                    merge_video_audio()
                    st.success("✅ 视频合成完成！")

                st.success("🎉 最终配音视频已生成！")
                if os.path.exists(DUB_VIDEO):
                    size_mb = os.path.getsize(DUB_VIDEO) / (1024 * 1024)
                    st.info(f"📁 输出文件: `{DUB_VIDEO}` ({size_mb:.1f} MB)")

                st.rerun()
            except Exception as e:
                st.error(f"❌ 合并/合成失败: {e}")

        # ── 显示已生成的视频（手机屏幕大小） ──
        if dub_video_exists:
            size_mb = os.path.getsize(DUB_VIDEO) / (1024 * 1024)
            st.markdown(f"**📺 预览输出视频** (`{DUB_VIDEO}`, {size_mb:.1f} MB)")
            try:
                with open(DUB_VIDEO, "rb") as vf:
                    video_bytes = vf.read()
                phone_col1, phone_col2, phone_col3 = st.columns([1, 1, 1])
                with phone_col2:
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"无法加载视频预览: {e}")

    st.markdown("---")

    # ── 逐片段列表 ──
    st.subheader("📜 配音片段列表")

    # 筛选选项
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        status_filter = st.selectbox(
            "筛选状态",
            options=["全部", "已配音", "未配音"],
            index=0,
        )
    with filter_col2:
        # 搜索框
        search_text = st.text_input("🔍 搜索片段内容", placeholder="输入原文或译文关键词...")

    # 加载 SRT 条目（缓存），供下方保存按钮和展示循环使用
    src_srt_entries = _cached_load_src_srt()
    trans_srt_entries = _cached_load_trans_srt()
    # 每个配音行绑定的角色（文本匹配，与"按角色批量分配音色"一致）
    _, role_by_key, _ = build_line_role_map(df, trans_srt_entries)

    # ── 保存全部按钮 ──
    save_all_col1, save_all_col2 = st.columns([3, 1])
    with save_all_col1:
        st.caption("💡 修改译文或参考选择后，点击右侧按钮一次性保存所有更改。")
    with save_all_col2:
        if st.button("💾 保存全部译文 & 参考选择", type="primary", use_container_width=True, key="save_all_btn"):
            try:
                saved_count = 0
                for idx, row in df.iterrows():
                    number = row['number']
                    lines_list = parse_lines(row['lines'])
                    ref_lines_old = parse_lines(row.get('ref_lines', ''))
                    instruct_lines_old = parse_lines(row.get('instruct_lines', ''))
                    ref_files_old = parse_lines(row.get('ref_files', ''))

                    # 获取该片段在 trans.srt 中匹配的译文文本（作为默认值）
                    seg_start_srt = row.get('start_time', '')
                    seg_end_srt = row.get('end_time', '')
                    matched_trans_texts = get_srt_texts_by_segment(trans_srt_entries, seg_start_srt, seg_end_srt)
                    num_lines = len(matched_trans_texts) if matched_trans_texts else len(lines_list)

                    edited_lines = []
                    new_ref_lines = []
                    new_instruct_lines = []
                    new_ref_files = []
                    for i in range(num_lines):
                        # 回退到 Excel 中最后保存的值，绝不回退到 SRT 原始文本
                        saved_text = lines_list[i] if i < len(lines_list) else ''
                        trans_val = st.session_state.get(f"trans_{number}_{i}", saved_text)
                        ref_val = st.session_state.get(f"ref_line_{number}_{i}", ref_lines_old[i] if i < len(ref_lines_old) else '1')
                        inst_val = st.session_state.get(f"inst_line_{number}_{i}", instruct_lines_old[i] if i < len(instruct_lines_old) else '')
                        rf_val = st.session_state.get(f"ref_file_{number}_{i}", ref_files_old[i] if i < len(ref_files_old) else '')
                        edited_lines.append(trans_val)
                        new_ref_lines.append(ref_val)
                        new_instruct_lines.append(inst_val)
                        new_ref_files.append(rf_val)

                    df.at[idx, 'lines'] = str(edited_lines)
                    df.at[idx, 'ref_lines'] = str(new_ref_lines)
                    df.at[idx, 'instruct_lines'] = str(new_instruct_lines)
                    df.at[idx, 'ref_files'] = str(new_ref_files)
                    saved_count += 1

                df.to_excel(TASKS_FILE, index=False)
                _cached_load_tasks.clear()
                st.success(f"✅ 已保存全部 {saved_count} 个片段的译文和参考选择！")
            except Exception as e:
                st.error(f"❌ 保存失败: {e}")
    st.markdown("---")

    # 遍历每个片段
    found_any = False
    # src_srt_entries / trans_srt_entries 已在上方加载
    # 当前模式下每行可选的音频/音色列表
    refs_files = list_refs_dir() if qwen_mode == 2 else []
    persona_names = list_personas(get_persona_dir()) if qwen_mode == 3 else []

    for idx, row in df.iterrows():
        number = row['number']
        lines_list = parse_lines(row['lines'])
        cut_off = row.get('cut_off', 0)
        ref_lines = parse_lines(row.get('ref_lines', ''))
        if not ref_lines or len(ref_lines) != len(lines_list):
            ref_lines = ['1'] * len(lines_list) if lines_list else []
        instruct_lines = parse_lines(row.get('instruct_lines', ''))
        if not instruct_lines or len(instruct_lines) != len(lines_list):
            instruct_lines = [''] * len(lines_list) if lines_list else []
        ref_files = parse_lines(row.get('ref_files', ''))
        if not ref_files or len(ref_files) != len(lines_list):
            ref_files = [''] * len(lines_list) if lines_list else []

        # 从 SRT 文件按时间戳匹配原文和译文（以 start_time 和 end_time 为时间区间）
        # 使用左开右闭区间 (seg_start < end_sec <= seg_end) 避免相邻片段的边界重复匹配
        seg_start_srt = row.get('start_time', '')
        seg_end_srt = row.get('end_time', '')
        matched_src_texts = get_srt_texts_by_segment(src_srt_entries, seg_start_srt, seg_end_srt)
        matched_trans_texts = get_srt_texts_by_segment(trans_srt_entries, seg_start_srt, seg_end_srt)

        # 检查音频状态
        all_generated, audio_paths, combined_audio = get_segment_status(number, lines_list)

        # 应用筛选
        if status_filter == "已配音" and not all_generated:
            continue
        if status_filter == "未配音" and all_generated:
            continue

        # 应用搜索（同时搜索 SRT 匹配的原文和 SRT 匹配的译文）
        if search_text:
            src_text = ' '.join(t for t, _ in matched_src_texts) if matched_src_texts else ''
            trans_text = ' '.join(t for t, _ in matched_trans_texts) if matched_trans_texts else ''
            if search_text.lower() not in src_text.lower() and search_text.lower() not in trans_text.lower():
                continue

        found_any = True

        # ── 片段卡片 ──
        with st.container(border=True):
            # 标题行
            header_col1, header_col2, header_col3, header_col4 = st.columns([1, 2, 2, 2])

            with header_col1:
                chunk_info = ""
                if cut_off == 1:
                    chunk_info = " 🔪切分点"
                st.markdown(f"**片段 #{number}**{chunk_info}")

            with header_col2:
                dur = row.get('duration', 0)
                st.markdown(f"⏱ **最长音频**: {format_time_display(dur)}")

            with header_col3:
                if all_generated:
                    st.markdown("✅ **已配音**")
                else:
                    st.markdown("⏳ **未配音**")

            with header_col4:
                if cut_off == 1:
                    st.markdown("🏁 **块结束**")
                else:
                    st.markdown("")

            # 原文 & 译文 预览（可展开）
            with st.expander("📄 查看原文 & 译文", expanded=not all_generated):
                col_src, col_trans = st.columns(2)

                with col_src:
                    st.markdown("**🔤 原文 (从 SRT 按时间匹配):**")
                    if matched_src_texts:
                        for i, (sl, role) in enumerate(matched_src_texts):
                            badge = f" 🎭 {role}" if role else ""
                            st.markdown(f">  [{i + 1}]{badge} {sl}")
                    else:
                        st.caption("(空)")

                with col_trans:
                    st.markdown("**🌐 译文 (从 SRT 按时间匹配，可编辑):**")
                    if matched_trans_texts:
                        for i, (ll, role) in enumerate(matched_trans_texts):
                            if qwen_mode == 1:
                                line_cols = st.columns([3, 2])
                            else:
                                line_cols = st.columns([3, 1, 2])
                            with line_cols[0]:
                                # 优先级：session_state（编辑中）> Excel lines_list（已保存）> SRT 匹配文本（原始）
                                saved_val = lines_list[i] if i < len(lines_list) else ''
                                display_val = saved_val or ll
                                default_val = st.session_state.get(f"trans_{number}_{i}", display_val)
                                # 徽章角色与"按角色批量分配"一致（文本匹配，按 Excel 行号）；
                                # 未绑定行回退到 SRT 时间匹配行自带的角色，保证每行都显示
                                role_display = role_by_key.get((int(number), i)) or role
                                if role_display:
                                    st.caption(f"🎭 {role_display}")
                                st.text_input(
                                    f"第 {i + 1} 行",
                                    value=default_val,
                                    key=f"trans_{number}_{i}",
                                    label_visibility="collapsed",
                                    placeholder="输入译文...",
                                )
                            if qwen_mode == 1:
                                with line_cols[1]:
                                    st.caption(f"行{i+1} 原角色参考")
                            else:
                                with line_cols[1]:
                                    line_sel = ref_files[i] if i < len(ref_files) else ''
                                    if qwen_mode == 2:
                                        opts = [''] + refs_files
                                        label_map = {o: ("默认固定参考" if o == '' else o) for o in opts}
                                    else:
                                        opts = [''] + persona_names
                                        label_map = {o: ("默认音色" if o == '' else o) for o in opts}
                                    if line_sel not in opts:
                                        line_sel = ''
                                    st.selectbox(
                                        f"选择{i+1}",
                                        options=opts,
                                        format_func=lambda x: label_map.get(x, x),
                                        index=opts.index(line_sel),
                                        key=f"ref_file_{number}_{i}",
                                        label_visibility="collapsed",
                                    )
                            instruct_col_idx = 2 if qwen_mode != 1 else 1
                            with line_cols[instruct_col_idx]:
                                line_instruct = instruct_lines[i] if i < len(instruct_lines) else ''
                                st.text_input(
                                    f"指令{i+1}",
                                    value=st.session_state.get(f"inst_line_{number}_{i}", line_instruct),
                                    key=f"inst_line_{number}_{i}",
                                    label_visibility="collapsed",
                                    placeholder="情感指令(可选)",
                                    help="如: 带点哭腔、用温柔的语气",
                                )

                        # ── 片段级独立保存按钮 ──
                        st.markdown("---")
                        save_col1, save_col2 = st.columns([1, 3])
                        with save_col1:
                            if st.button(f"💾 保存片段 #{number}", key=f"save_seg_{number}", use_container_width=True):
                                try:
                                    edited_lines = []
                                    new_ref_lines = []
                                    new_instruct_lines = []
                                    new_ref_files = []
                                    for i in range(len(matched_trans_texts)):
                                        trans_key = f"trans_{number}_{i}"
                                        ref_key = f"ref_line_{number}_{i}"
                                        inst_key = f"inst_line_{number}_{i}"
                                        rf_key = f"ref_file_{number}_{i}"
                                        trans_val = st.session_state.get(trans_key, matched_trans_texts[i][0])
                                        ref_val = st.session_state.get(ref_key, ref_lines[i] if i < len(ref_lines) else '1')
                                        inst_val = st.session_state.get(inst_key, instruct_lines[i] if i < len(instruct_lines) else '')
                                        rf_val = st.session_state.get(rf_key, ref_files[i] if i < len(ref_files) else '')
                                        edited_lines.append(trans_val)
                                        new_ref_lines.append(ref_val)
                                        new_instruct_lines.append(inst_val)
                                        new_ref_files.append(rf_val)
                                    df.at[idx, 'lines'] = str(edited_lines)
                                    df.at[idx, 'ref_lines'] = str(new_ref_lines)
                                    df.at[idx, 'instruct_lines'] = str(new_instruct_lines)
                                    df.at[idx, 'ref_files'] = str(new_ref_files)
                                    df.to_excel(TASKS_FILE, index=False)
                                    _cached_load_tasks.clear()
                                    st.success(f"✅ 片段 #{number} 译文已保存！")
                                except Exception as e:
                                    st.error(f"❌ 保存失败: {e}")
                        with save_col2:
                            st.caption("保存后刷新，译文将持久化")
                    else:
                        st.caption("(空)")

            # 操作区
            op_col1, op_col2, op_col3, op_col4, op_col5, op_col6 = st.columns([1, 1, 1, 2, 2, 2])

            with op_col1:
                # 配音（首次 / 覆盖）
                dub_key = f"dub_{number}"
                if st.button(f"🎤 配音", key=dub_key, use_container_width=True):
                    if not lines_list:
                        st.warning(f"片段 #{number} 没有需要配音的文本。")
                    else:
                        with st.spinner(f"正在为片段 #{number} 生成音频 ({len(lines_list)} 行)..."):
                            ok = redub_single_segment(number, lines_list, df, ref_lines=ref_lines, instruct_lines=instruct_lines, ref_files=ref_files)
                            if ok:
                                _invalidate_audio_cache(number)
                                st.success(f"✅ 片段 #{number} 配音完成！")
                                st.rerun()
                            else:
                                st.error(f"❌ 片段 #{number} 配音失败。")

            with op_col2:
                # 重新配音（带确认）
                redub_key = f"redub_{number}"
                with st.popover("🔄 重新配音", use_container_width=True):
                    st.markdown(f"**确认重新配音片段 #{number}？**")
                    if qwen_mode == 1:
                        ref_summary = ', '.join([f"行{i+1}:参考{r}" for i, r in enumerate(ref_lines)])
                    else:
                        ref_summary = ', '.join([f"行{i+1}:{r if r else '默认'}" for i, r in enumerate(ref_files)])
                    st.caption(f"将删除旧的音频文件并重新生成 {len(lines_list)} 行。参考: {ref_summary}")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ 确认", key=f"redub_confirm_{number}", use_container_width=True):
                            if not lines_list:
                                st.warning(f"片段 #{number} 没有需要配音的文本。")
                            else:
                                with st.spinner(f"正在重新配音片段 #{number} ({len(lines_list)} 行)..."):
                                    ok = redub_single_segment(number, lines_list, df, ref_lines=ref_lines, instruct_lines=instruct_lines, ref_files=ref_files)
                                    if ok:
                                        _invalidate_audio_cache(number)
                                        st.success(f"✅ 片段 #{number} 重新配音完成！")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 重新配音失败。")
                    with col_no:
                        if st.button("❌ 取消", key=f"redub_cancel_{number}", use_container_width=True):
                            pass

            with op_col3:
                # 删除配音（带确认）
                del_key = f"del_{number}"
                with st.popover("🗑️ 删除配音", use_container_width=True):
                    st.markdown(f"**确认删除片段 #{number} 的音频？**")
                    st.caption("将删除该片段的所有临时文件和最终音频文件。")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ 确认删除", key=f"del_confirm_{number}", use_container_width=True):
                            deleted = delete_segment_audio(number, lines_list)
                            _invalidate_audio_cache(number)
                            if deleted > 0:
                                st.success(f"✅ 已删除 {deleted} 个音频文件")
                            else:
                                st.info("没有找到需要删除的音频文件")
                            st.rerun()
                    with col_no:
                        if st.button("❌ 取消", key=f"del_cancel_{number}", use_container_width=True):
                            pass

            with op_col4:
                # 配音音频播放器
                st.caption("🎤 配音")
                if all_generated and combined_audio:
                    audio_html = get_audio_player(combined_audio)
                    st.markdown(audio_html, unsafe_allow_html=True)
                elif audio_paths and any(p is not None for p in audio_paths):
                    # 部分生成时，尝试播放已存在的部分
                    try:
                        partial_audio = AudioSegment.empty()
                        for ap in audio_paths:
                            if ap and os.path.exists(ap):
                                partial_audio += AudioSegment.from_wav(ap)
                        if len(partial_audio) > 0:
                            buf = BytesIO()
                            partial_audio.export(buf, format="wav")
                            buf.seek(0)
                            audio_html = get_audio_player(buf)
                            st.markdown(audio_html, unsafe_allow_html=True)
                        else:
                            st.caption("无音频")
                    except Exception:
                        st.caption("无音频")
                else:
                    st.caption("暂无音频")

            with op_col5:
                # 原音音频播放器
                st.caption("🔊 原音")
                try:
                    orig_audio_buf = get_original_audio_segment(
                        number,
                        row.get('start_time', ''),
                        row.get('end_time', '')
                    )
                    if orig_audio_buf:
                        audio_html = get_audio_player(orig_audio_buf)
                        st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.caption("无原音")
                except Exception:
                    st.caption("无原音")

            with op_col6:
                # 显示每行音频时长信息（带 session_state 缓存，避免重复运行 ffmpeg）
                if all_generated and audio_paths:
                    dur_cache_key = f"_audio_durations_{number}"
                    cached_durs = st.session_state.get(dur_cache_key)
                    if cached_durs is not None and len(cached_durs) == len(audio_paths):
                        durations = cached_durs
                    else:
                        durations = []
                        for ap in audio_paths:
                            if ap and os.path.exists(ap):
                                d = get_audio_duration(ap)
                                durations.append(d)
                        st.session_state[dur_cache_key] = durations
                    total_dur = sum(durations)
                    dur_str = " + ".join(f"{d:.2f}s" for d in durations)
                    st.caption(f"📊 各句时长: {dur_str} = **{total_dur:.2f}s**")
                elif lines_list:
                    st.caption(f"📝 待配音: {len(lines_list)} 行")
                else:
                    st.caption("")

        # 片段间间隔
        st.markdown("<br>", unsafe_allow_html=True)

    if not found_any:
        st.info("没有匹配当前筛选条件的片段。")
        if status_filter != "全部":
            st.markdown(f"当前筛选: **{status_filter}**。尝试切换到「全部」查看所有片段。")

    # ── 底部提示 ──
    st.markdown("---")
    st.caption(
        "💡 **操作说明**: "
        "🎤 顶部选择三种配音方式之一：原角色音频参考 / 固定音频参考 / 固定音色；"
        "「固定音频参考」先在音频库批量上传音频，「固定音色」可在音色库新增/试听固化音色并批量分配；"
        "展开「查看原文 & 译文」后，每行可独立选择音频/音色并填写情感指令；"
        "🎤「配音」直接生成/覆盖音频（使用各行已选音频/音色与情感指令）；"
        "🔄「重新配音」先删旧文件再重新生成（需确认）；"
        "🗑️「删除配音」清除该片段所有音频文件（需确认）。"
        "修改译文或选择后，点击「💾 保存译文 & 参考选择」保存。"
        "🎬「将最终音频合并到视频中」合并所有配音片段并烧录字幕到原始视频，生成 `output/output_dub.mp4`。"
    )


if __name__ == "__main__":
    main()
