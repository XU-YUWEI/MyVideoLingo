import os, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import numpy as np
import platform
from core.utils import *

# 修改变量定义部分
SRC_FONT_SIZE = 15
#TRANS_FONT_SIZE = 17

# 从配置文件读取，如果没有则使用默认值
SRC_FONT_NAME = 'Arial'
#TRANS_FONT_NAME = load_key("subtitle.font") or 'Arial'

# 如果是 Linux/Mac，保持你原有的逻辑覆盖（最小化修改）
#if platform.system() == 'Linux':
#    FONT_NAME = 'NotoSansCJK-Regular'
#    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
#elif platform.system() == 'Darwin':
#    FONT_NAME = 'Arial Unicode MS'
 #   TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'

# 修改这里：动态读取翻译字幕颜色
#TRANS_FONT_COLOR = load_key("subtitle.trans_color") or '&H00FFFF' 

TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"
    
def check_gpu_available():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False

def merge_subtitles_to_video():
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
    # -------------------------------------------
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    # 获取源视频的编码格式，尽量保持一致性
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,bit_rate',
        '-of', 'csv=p=0', video_file
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

    ffmpeg_cmd = [
        'ffmpeg', '-i', video_file,
        '-vf', (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"subtitles={SRC_SRT}:force_style='FontSize={SRC_FONT_SIZE},FontName={SRC_FONT_NAME},"
            f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1',"
            f"subtitles={TRANS_SRT}:force_style='FontSize={trans_font_size},FontName={font_name},"
            f"PrimaryColour={trans_font_color},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle={trans_border_style}'"
        ).encode('utf-8'),
    ]

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration (NVENC) with high quality settings.[/bold green]")
        ffmpeg_cmd.extend([
            '-c:v', 'h264_nvenc',
            '-cq', '17',          # 恒定质量模式，值越低质量越高（0-51），17为高质量
            '-preset', 'p7',      # p7 是 NVENC 最高质量预设
            '-rc', 'vbr',         # 可变码率
            '-b:v', '50M',        # 最大码率 50 Mbps
            '-maxrate', '80M',    # 峰值码率 80 Mbps
            '-bufsize', '80M',    # 缓冲区大小
            '-pix_fmt', 'yuv420p'
        ])
    else:
        rprint("[bold green]Using CPU encoding with high quality settings (libx264 CRF 17).[/bold green]")
        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-crf', '17',         # CRF 17 = 视觉无损（0-51，越低越好，0=无损）
            '-preset', 'slow',    # slow 预设提供更好的压缩效率
            '-pix_fmt', 'yuv420p'
        ])
        # 如果能获取到源视频码率，使用其作为最大码率限制
        if src_bitrate and src_bitrate != 'N/A' and src_bitrate.isdigit():
            ffmpeg_cmd.extend(['-maxrate', f'{int(src_bitrate)//1000}k'])
    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint("\n❌ FFmpeg execution error")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    merge_subtitles_to_video()