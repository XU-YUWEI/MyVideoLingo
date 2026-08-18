import datetime
import re
import pandas as pd
from core._8_1_audio_task import time_diff_seconds
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *
from core.utils.models import *

SRC_SRT = "output/src.srt"
TRANS_SRT = "output/trans.srt"
MAX_MERGE_COUNT = 5
ESTIMATOR = None

def calc_if_too_fast(est_dur, tol_dur, duration, tolerance):
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
        return 2
    elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
        return 1
    elif est_dur < duration - tolerance:  # Speaking speed too slow
        return -1
    else:  # Normal speaking speed
        return 0

def merge_rows(df, start_idx, merge_count):
    """Merge multiple rows and calculate cumulative values"""
    merged = {
        'est_dur': df.iloc[start_idx]['est_dur'],
        'tol_dur': df.iloc[start_idx]['tol_dur'],
        'duration': df.iloc[start_idx]['duration']
    }
    
    while merge_count < MAX_MERGE_COUNT and (start_idx + merge_count) < len(df):
        next_row = df.iloc[start_idx + merge_count]
        merged['est_dur'] += next_row['est_dur']
        merged['tol_dur'] += next_row['tol_dur']
        merged['duration'] += next_row['duration']
        
        speed_flag = calc_if_too_fast(
            merged['est_dur'],
            merged['tol_dur'],
            merged['duration'],
            df.iloc[start_idx + merge_count]['tolerance']
        )
        
        if speed_flag <= 0 or merge_count == 2:
            df.at[start_idx + merge_count, 'cut_off'] = 1
            return merge_count + 1
        
        merge_count += 1
    
    # If no suitable merge point is found
    if merge_count >= MAX_MERGE_COUNT or (start_idx + merge_count) >= len(df):
        df.at[start_idx + merge_count - 1, 'cut_off'] = 1
    return merge_count

def analyze_subtitle_timing_and_speed(df):
    rprint("[🔍 Analyzing] Calculating subtitle timing and speed...")
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    TOLERANCE = load_key("tolerance")
    whole_dur = get_audio_duration(_RAW_AUDIO_FILE)
    df['gap'] = 0.0  # Initialize gap column
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(df.loc[i, 'end_time'], '%H:%M:%S.%f').time()
        next_start = datetime.datetime.strptime(df.loc[i + 1, 'start_time'], '%H:%M:%S.%f').time()
        df.loc[i, 'gap'] = time_diff_seconds(current_end, next_start, datetime.date.today())
    
    # Set the gap for the last line
    last_end = datetime.datetime.strptime(df.iloc[-1]['end_time'], '%H:%M:%S.%f').time()
    last_end_seconds = (last_end.hour * 3600 + last_end.minute * 60 + 
                       last_end.second + last_end.microsecond / 1000000)
    df.iloc[-1, df.columns.get_loc('gap')] = whole_dur - last_end_seconds
    
    df['tolerance'] = df['gap'].apply(lambda x: TOLERANCE if x > TOLERANCE else x)
    df['tol_dur'] = df['duration'] + df['tolerance']
    df['est_dur'] = df.apply(lambda x: estimate_duration(x['text'], ESTIMATOR), axis=1)

    ## Calculate speed indicators
    accept = load_key("speed_factor.accept") # Maximum acceptable speed factor
    def calc_if_too_fast(row):
        est_dur = row['est_dur']
        tol_dur = row['tol_dur']
        duration = row['duration']
        tolerance = row['tolerance']
        
        if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
            return 2
        elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
            return 1
        elif est_dur < duration - tolerance:  # Speaking speed too slow
            return -1
        else:  # Normal speaking speed
            return 0
    
    df['if_too_fast'] = df.apply(calc_if_too_fast, axis=1)
    return df

def process_cutoffs(df):
    rprint("[✂️ Processing] Generating cutoff points...")
    df['cut_off'] = 0  # Initialize cut_off column
    df.loc[df['gap'] >= load_key("tolerance"), 'cut_off'] = 1  # Set to 1 when gap is greater than TOLERANCE
    idx = 0
    while idx < len(df):
        # Process marked split points
        if df.iloc[idx]['cut_off'] == 1:
            if df.iloc[idx]['if_too_fast'] == 2:
                rprint(f"[⚠️ Warning] Line {idx} is too fast and cannot be fixed by speed adjustment")
            idx += 1
            continue

        # Process the last line
        if idx + 1 >= len(df):
            df.at[idx, 'cut_off'] = 1
            break

        # Process normal or slow lines
        if df.iloc[idx]['if_too_fast'] <= 0:
            if df.iloc[idx + 1]['if_too_fast'] <= 0:
                df.at[idx, 'cut_off'] = 1
                idx += 1
            else:
                idx += merge_rows(df, idx, 1)
        # Process fast lines
        else:
            idx += merge_rows(df, idx, 1)
    
    return df

def resolve_persona(speaker):
    """根据行首角色标记解析固化音色名。
    优先级: config 中已合并的 Excel 覆盖 > qwen3_tts.persona_map > qwen3_tts.fallback_persona。
    无标记 / 未映射 一律回退 fallback_persona（默认旁白）。"""
    try:
        persona_map = load_key("qwen3_tts.persona_map") or {}
        fallback = load_key("qwen3_tts.fallback_persona") or ""
    except KeyError:
        persona_map, fallback = {}, ""
    if speaker and speaker in persona_map:
        return persona_map[speaker]
    return fallback


def resolve_instruct(speaker):
    """根据行首角色标记解析情感指令。
    优先级: config 中已合并的 Excel 覆盖 > qwen3_tts.instruct_map。
    无标记 / 未映射 返回空串（调用侧回落全局 qwen3_tts.instruct）。"""
    try:
        instruct_map = load_key("qwen3_tts.instruct_map") or {}
    except KeyError:
        instruct_map = {}
    if speaker and speaker in instruct_map:
        return instruct_map[speaker] or ""
    return ""


def gen_dub_chunks():
    rprint("[🎬 Starting] Generating dubbing chunks...")
    df = pd.read_excel(_8_1_AUDIO_TASK)
    
    rprint("[📊 Processing] Analyzing timing and speed...")
    df = analyze_subtitle_timing_and_speed(df)
    
    rprint("[✂️ Processing] Processing cutoffs...")
    df = process_cutoffs(df)

    rprint("[📝 Reading] Loading transcript files...")
    # 解析 SRT 并保留时间与角色标记，用于按片段时间区间匹配行
    def parse_srt_with_time(path):
        entries = []  # (start_sec, end_sec, text, speaker)
        for block in open(path, "r", encoding="utf-8").read().strip().split('\n\n'):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if len(lines) >= 3:
                m = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1])
                if not m:
                    continue
                def to_sec(g):
                    return int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
                start = to_sec(m.groups()[:4])
                end = to_sec(m.groups()[4:])
                text = ' '.join(lines[2:])
                text = re.sub(r'\([^)]*\)|（[^）]*）', '', text).strip().replace('-', '')
                # 提取行首 [角色N] 标记并剥离，避免 TTS 读出；无标记则 speaker 为 None
                speaker = None
                tag = re.match(r'^\[([^\]]*)\]\s*', text)
                if tag:
                    speaker = tag.group(1).strip()
                    text = text[tag.end():]
                entries.append((start, end, text, speaker))
        return entries

    def str_time_to_sec(t):
        parts = str(t).split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    trans_entries = parse_srt_with_time(TRANS_SRT)
    ori_entries = parse_srt_with_time(SRC_SRT)
    content_lines = [t for _, _, t, _ in trans_entries]
    ori_content_lines = [t for _, _, t, _ in ori_entries]
    speaker_lines = [s for _, _, _, s in trans_entries]
    total_split = len(content_lines)

    # 按时间匹配分配：取行中点落在片段时间区间内的 SRT 行（中点唯一归属，避免相邻片段重复匹配）
    rprint("[🔗 Processing] Assigning subtitle lines by timestamp...")
    df['lines'] = None
    df['src_lines'] = None
    df['persona_lines'] = None
    df['instruct_lines'] = None

    line_idx = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        s0 = str_time_to_sec(row['start_time'])
        e0 = str_time_to_sec(row['end_time'])
        chunk_trans = [t for a, b, t, _ in trans_entries if s0 <= (a + b) / 2 < e0]
        chunk_ori = [t for a, b, t, _ in ori_entries if s0 <= (a + b) / 2 < e0]
        chunk_spk = [s for a, b, _, s in trans_entries if s0 <= (a + b) / 2 < e0]
        if not chunk_trans and line_idx < total_split:
            # 兜底：时间匹配失败时按顺序取下一行，保证行数不缺失
            chunk_trans = [content_lines[line_idx]]
            chunk_ori = [ori_content_lines[line_idx]] if line_idx < len(ori_content_lines) else ['']
            chunk_spk = [speaker_lines[line_idx]] if line_idx < len(speaker_lines) else [None]
            line_idx += 1
        df.at[idx, 'lines'] = chunk_trans
        df.at[idx, 'src_lines'] = chunk_ori
        df.at[idx, 'persona_lines'] = [resolve_persona(s) for s in chunk_spk]
        df.at[idx, 'instruct_lines'] = [resolve_instruct(s) for s in chunk_spk]

    # Save results
    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[✅ Complete] Lines assigned successfully!")

if __name__ == "__main__":
    gen_dub_chunks()