import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel

# Constants
SETTINGS_FILE = 'batch/tasks_setting.xlsx'
INPUT_FOLDER = os.path.join('batch', 'input')
VALID_DUBBING_VALUES = [0, 1]
# 与 config.yaml 中 allowed_video_formats 保持一致
VIDEO_EXTENSIONS = ('mp4', 'mov')

console = Console()


def is_video_file(filename: str) -> bool:
    """判断是否为受支持格式的视频文件（排除 jpg/png 等缩略图辅助文件）"""
    ext = os.path.splitext(filename)[1][1:].lower()
    return ext in VIDEO_EXTENSIONS


def is_done(row: pd.Series) -> bool:
    """该任务是否已完成（字幕 Status 或配音 DubbingStatus 为 Done）"""
    for col in ('Status', 'DubbingStatus'):
        val = row.get(col)
        if not pd.isna(val) and str(val) == 'Done':
            return True
    return False


def check_settings():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    df = pd.read_excel(SETTINGS_FILE)
    # 仅统计视频文件，忽略下载时顺带产生的缩略图（.jpg 等）
    input_files = set(f for f in os.listdir(INPUT_FOLDER) if is_video_file(f))
    excel_files = set(df['Video File'].tolist())
    files_not_in_excel = input_files - excel_files

    all_passed = True
    local_video_tasks = 0
    url_tasks = 0

    if files_not_in_excel:
        console.print(Panel(
            "\n".join([f"- {file}" for file in files_not_in_excel]),
            title="[bold yellow]Warning: Files in input folder not mentioned in Excel sheet",
            expand=False
        ))

    for index, row in df.iterrows():
        # 已完成的任务（字幕/配音 Done）不再参与校验
        if is_done(row):
            continue

        video_file = row['Video File']
        source_language = row['Source Language']
        dubbing = row['Dubbing']

        if video_file.startswith('http'):
            url_tasks += 1
        elif os.path.isfile(os.path.join(INPUT_FOLDER, video_file)):
            local_video_tasks += 1
        else:
            # 文件缺失仅警告，不阻塞整个批次；单行失败由主流程标记 Error 后继续
            console.print(Panel(f"Invalid video file or URL 「{video_file}」", title=f"[bold yellow]Warning in row {index + 2}", expand=False))

        if not pd.isna(dubbing):
            if int(dubbing) not in VALID_DUBBING_VALUES:
                console.print(Panel(f"Invalid dubbing value 「{dubbing}」", title=f"[bold red]Error in row {index + 2}", expand=False))
                all_passed = False

    if all_passed:
        console.print(Panel(f"✅ All settings passed the check!\nDetected {local_video_tasks} local video tasks and {url_tasks} URL tasks.", title="[bold green]Success", expand=False))

    return all_passed


if __name__ == "__main__":  
    check_settings()