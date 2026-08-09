import pandas as pd
import os
import re
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def build_speaker_mapping(df_text):
    """收集 df_text 中所有说话人标签，按序映射为 角色1/角色2/...（SPEAKER_00 → 角色1）"""
    if 'speaker_id' not in df_text.columns:
        return {}, []
    labels = sorted(df_text['speaker_id'].dropna().astype(str).unique())
    return {lab: f"角色{i + 1}" for i, lab in enumerate(labels)}, labels


def get_sentence_speaker(df_text, start, end):
    """统计 [start, end] 区间内 word 的重叠时长，取重叠最长的说话人标签；无则返回 None"""
    if 'speaker_id' not in df_text.columns:
        return None
    mask = (df_text['end'] > start) & (df_text['start'] < end)
    seg = df_text.loc[mask, ['start', 'end', 'speaker_id']].dropna(subset=['speaker_id'])
    if seg.empty:
        return None
    overlap = {}
    for _, r in seg.iterrows():
        ov = min(end, r['end']) - max(start, r['start'])
        if ov > 0:
            lab = str(r['speaker_id'])
            overlap[lab] = overlap.get(lab, 0.0) + ov
    if not overlap:
        return None
    return max(overlap, key=overlap.get)


def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    
    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(word.lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        clean_sentence = remove_punctuation(sentence.lower()).replace(" ", "")
        sentence_len = len(clean_sentence)
        
        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos:current_pos+sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1
            
        if not match_found:
            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, 
                          full_words_str[current_pos:current_pos+len(clean_sentence)])
            print("\nOriginal sentence:", df_sentences['Source'][idx])
            raise ValueError("❎ No match found for sentence.")
    
    return time_stamp_list

def fix_display_timing(timestamps, pad=0.1, min_dur=1.2):
    """修正显示字幕时间轴：消重叠 + 最小时长 + 时间微调。

    - 时间微调: Whisper 词级时间戳普遍略偏晚，每句开始提前 pad、结束延后 pad
    - 消重叠: 当前句结束不晚于下一句开始，开始不早于上一句结束
    - 最小时长: 过短的字幕向后延长到 min_dur（不超过下一句开始）
    返回新的 (start, end) 列表（单位：秒）。
    """
    result = []
    for i, (s, e) in enumerate(timestamps):
        s = max(0.0, s - pad)
        e = e + pad
        # 消重叠：开始不早于上一句结束
        if result:
            s = max(s, result[-1][1])
        # 消重叠：结束不晚于下一句开始
        next_start = timestamps[i + 1][0] if i + 1 < len(timestamps) else e
        e = min(e, next_start)
        # 最小时长：过短则向后延长（不越过下一句开始）
        if e - s < min_dur:
            e = min(s + min_dur, next_start)
        result.append((s, e))
    return result


def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # 按时间范围统计每句的说话人 → 角色N（没有说话人不加前缀）
    speaker_map, _ = build_speaker_mapping(df_text)
    df_trans_time['speaker'] = [
        speaker_map.get(get_sentence_speaker(df_text, s, e)) for s, e in time_stamp_list
    ]

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # 显示字幕时间轴修正：消重叠 + 最小时长 + 时间微调（仅用于显示字幕，配音字幕保持原始时间轴）
    if for_display:
        pad = load_key("subtitle.time_pad")
        min_dur = load_key("subtitle.min_display_duration")
        df_trans_time['timestamp'] = fix_display_timing(df_trans_time['timestamp'].tolist(), pad=pad, min_dur=min_dur)

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', x).strip())

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        def _fmt(row, col):
            text = str(row[col]).strip()
            spk = row.get('speaker')
            if spk and text:
                text = f"[{spk}] {text}"
            return text
        return ''.join([f"{i+1}\n{row['timestamp']}\n{_fmt(row, columns[0])}\n{_fmt(row, columns[1]) if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()