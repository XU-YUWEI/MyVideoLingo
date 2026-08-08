import os
import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

#TRANS_FONT_SIZE = 17
#if platform.system() == 'Linux':
#    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
#if platform.system() == 'Darwin':
#    TRANS_FONT_NAME = 'Arial Unicode MS'

#TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
      # --- 核心修改：在函数内部实时读取最新的配置 ---
    font_name = load_key("subtitle.font") or 'Arial'
    trans_font_color = load_key("subtitle.trans_color") or '&H00FFFF'
    trans_font_size = load_key("subtitle.font_size") or 17
    
    # 如果开启背景则使用 BorderStyle=4，否则使用 BorderStyle=1 (仅边框)
    use_bg = load_key("subtitle.use_bg")
    trans_border_style = 4 if use_bg else 1
    
    # 针对不同系统的字体兼容性处理（保持原逻辑）
    if platform.system() == 'Linux':
        font_name = 'NotoSansCJK-Regular'
    elif platform.system() == 'Darwin':
        font_name = 'Arial Unicode MS'
        
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE
    
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={trans_font_size},"
        f"FontName={font_name},PrimaryColour={trans_font_color},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle={trans_border_style}'"
    )
    
    # 配音始终 100% 音量输出（无论是否存在背景音，不再压低配音）
    bg_volume = 1.0  # 背景音保持原始音量

    # 无 Demucs 分离产物（background.mp3）时，不使用背景音，配音（dub.mp3）直接作为输出音轨
    use_original_audio_as_bg = not os.path.exists(background_file)
    if use_original_audio_as_bg:
        rprint("[bold yellow]⚠️ background.mp3 不存在（未开启 Demucs 人声分离），配音直接作为输出音轨[/bold yellow]")
        cmd = [
            'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', normalized_dub_audio,
            '-filter_complex',
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
            f'{subtitle_filter}[v];'
            f'[1:a]anull[a]'
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
            '-filter_complex',
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
            f'{subtitle_filter}[v];'
            f'[1:a]volume={bg_volume}[bg];'
            f'[2:a]anull[dub];'
            f'[bg][dub]amix=inputs=2:duration=first:dropout_transition=3[a]'
        ]

    # 探查源视频编码信息，尽可能保持画质
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,bit_rate',
        '-of', 'csv=p=0', VIDEO_FILE
    ]
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        probe_output = probe_result.stdout.strip().split(',')
        src_codec = probe_output[0] if len(probe_output) > 0 else ''
        src_bitrate = probe_output[1] if len(probe_output) > 1 else ''
        rprint(f"[bold green]Source video codec: {src_codec}, bitrate: {src_bitrate}[/bold green]")
    except Exception as e:
        src_codec = ''
        src_bitrate = ''
        rprint(f"[bold yellow]Could not probe source video: {e}[/bold yellow]")

    # 以源视频码率为输出码率上限，避免输出文件体积远超源视频（原 GPU 50M/CRF17 会产生 GB 级文件）
    if src_bitrate and src_bitrate != 'N/A' and src_bitrate.isdigit():
        target_rate = src_bitrate          # 目标码率 = 源视频码率（bps）
        max_rate = src_bitrate             # 峰值码率上限 = 源视频码率
        buf_size = str(int(src_bitrate) * 2)  # VBV 缓冲 = 2×源视频码率
    else:
        target_rate = '2M'
        max_rate = '4M'
        buf_size = '8M'

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration (NVENC) with quality-based settings...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]',
            '-c:v', 'h264_nvenc',
            '-cq', '23',          # 恒定质量模式，值越低质量越高；23 为画质与体积平衡点
            '-preset', 'p7',      # NVENC 最高质量预设
            '-rc', 'vbr',         # 可变码率
            '-b:v', target_rate,  # 目标码率与源一致，防止输出文件过大
            '-maxrate', max_rate, # 峰值码率上限（源视频码率）
            '-bufsize', buf_size,
            '-pix_fmt', 'yuv420p'
        ])
    else:
        rprint("[bold green]Using CPU encoding with balanced settings (libx264 CRF 23)...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]',
            '-c:v', 'libx264',
            '-crf', '23',         # CRF 23 = 画质与体积平衡（原 17 接近无损，文件极大）
            '-preset', 'medium',  # medium 预设兼顾速度与压缩率
            '-maxrate', max_rate, # 峰值码率上限（源视频码率），避免输出文件过大
            '-bufsize', buf_size,
            '-pix_fmt', 'yuv420p'
        ])
    
    cmd.extend(['-c:a', 'aac', '-b:a', '192k', DUB_VIDEO])  # 提高音频码率至192k
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
