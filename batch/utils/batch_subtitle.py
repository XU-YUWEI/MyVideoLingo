import os
import gc
import shutil
import time
from functools import partial

# 在加载 pandas/torch 等重库之前先加载 onnxruntime：
# 若先 import pandas，onnxruntime_pybind11_state 的 DLL 会初始化失败（WinError 1114），
# 进而导致 whisper 步骤中 pyannote/torchmetrics 的导入全部失败
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from batch.utils.settings_check import check_settings
from core import *
from core._1_ytdlp import find_video_files
from core.utils.config_utils import load_key, update_key
from core.utils.onekeycleanup import cleanup, get_video_history_name

console = Console()

# ── 常量 ─────────────────────────────────────────────────
SETTINGS_FILE = 'batch/tasks_setting.xlsx'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
INPUT_DIR = 'batch/input'


def record_and_update_config(source_language: str, target_language: str):
    """暂存并切换当前任务的源语言与目标语言配置"""
    original_source_lang = load_key('whisper.language')
    original_target_lang = load_key('target_language')

    if source_language and not pd.isna(source_language):
        update_key('whisper.language', source_language)
    if target_language and not pd.isna(target_language):
        update_key('target_language', target_language)

    return original_source_lang, original_target_lang


def prepare_output_folder():
    """清空并重建 output 目录"""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)


def restore_from_error(video_file: str):
    """从 batch/output/ERROR/<视频名>/ 恢复中间产物到 output/"""
    video_name = get_video_history_name(os.path.splitext(video_file)[0])
    error_folder = os.path.join(ERROR_OUTPUT_DIR, video_name)

    if not os.path.exists(error_folder):
        console.print(f"[yellow]⚠️ 错误恢复目录不存在: {error_folder}[/yellow]")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for item in os.listdir(error_folder):
        src = os.path.join(error_folder, item)
        dst = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)

    console.print(f"[green]✅ 已从 ERROR 目录恢复 {video_file} 的中间文件[/green]")


def process_input_file(file: str):
    """处理输入文件：本地复制 或 YouTube 下载"""
    if file.startswith('http'):
        _1_ytdlp.download_video_ytdlp(file, resolution=load_key('ytb_resolution'))
        video_file = find_video_files()
    else:
        input_file = os.path.join(INPUT_DIR, file)
        output_file = os.path.join(OUTPUT_DIR, file)
        shutil.copy(input_file, output_file)
        video_file = output_file
    return {'video_file': video_file}


def split_sentences():
    """NLP 分句 → 语义分割"""
    _3_1_split_nlp.split_by_spacy()
    _3_2_split_meaning.split_sentences_by_meaning()


def summarize_and_translate():
    """摘要 → 翻译"""
    _4_1_summarize.get_summary()
    _4_2_translate.translate_all()


def process_and_align_subtitles():
    """字幕拆分 → 时间戳对齐"""
    _5_split_sub.split_for_sub_main()
    _6_gen_sub.align_timestamp_main()


def process_single_video(video_file: str, is_retry: bool = False):
    """
    对单个视频执行完整的字幕管线（不含配音）。
    返回 (success: bool, error_step: str, error_message: str)
    """
    if not is_retry:
        prepare_output_folder()

    steps = [
        ("🎥 复制/下载视频", partial(process_input_file, video_file)),
        ("🎙️ Whisper 语音识别", _2_asr.transcribe),
        ("✂️ NLP 分句 + 语义分割", split_sentences),
        ("📝 摘要 + 翻译", summarize_and_translate),
        ("⚡ 字幕拆分 + 时间戳对齐", process_and_align_subtitles),
        ("🎬 字幕嵌入视频", _7_sub_into_vid.merge_subtitles_to_video),
    ]

    current_step = ""
    for step_name, step_func in steps:
        current_step = step_name
        for attempt in range(3):
            try:
                console.print(Panel(
                    f"[bold green]{step_name}[/]",
                    subtitle=f"第 {attempt + 1}/3 次尝试" if attempt > 0 else None,
                    border_style="blue",
                ))
                result = step_func()
                if result is not None:
                    globals().update(result)
                break
            except Exception as e:
                if attempt == 2:
                    console.print(Panel(
                        f"[bold red]步骤 '{current_step}' 失败:[/]\n{str(e)}",
                        border_style="red",
                    ))
                    cleanup(ERROR_OUTPUT_DIR)
                    return False, current_step, str(e)
                console.print(Panel(
                    f"[yellow]第 {attempt + 1} 次尝试失败，正在重试…[/yellow]",
                    border_style="yellow",
                ))

    console.print(Panel("[bold green]🎉 字幕生成与嵌入全部完成！[/bold green]", border_style="green"))
    cleanup(SAVE_DIR)
    return True, "", ""


def process_batch_subtitle():
    """批量字幕处理主入口"""
    # ── 前置校验 ──
    if not check_settings():
        raise Exception("配置校验未通过，请检查 batch/tasks_setting.xlsx")

    df = pd.read_excel(SETTINGS_FILE)
    total = len(df)

    for index, row in df.iterrows():
        # 只处理 Status 为空 或 包含 Error 的行
        if pd.isna(row['Status']) or 'Error' in str(row['Status']):
            video_file = row['Video File']
            is_retry = not pd.isna(row['Status']) and 'Error' in str(row['Status'])

            # ── 标题面板 ──
            if is_retry:
                console.print(Panel(
                    f"重试失败任务: {video_file}\n任务 {index + 1}/{total}",
                    title="[bold yellow]🔄 重试任务",
                    expand=False,
                ))
                restore_from_error(video_file)
            else:
                console.print(Panel(
                    f"正在处理: {video_file}\n任务 {index + 1}/{total}",
                    title="[bold blue]📌 当前任务",
                    expand=False,
                ))

            # ── 语言切换 ──
            source_language = row['Source Language']
            target_language = row['Target Language']
            orig_src, orig_tgt = record_and_update_config(source_language, target_language)

            try:
                success, error_step, error_msg = process_single_video(video_file, is_retry)
                status_msg = "Done" if success else f"Error: {error_step} - {error_msg}"
            except Exception as e:
                status_msg = f"Error: 未捕获异常 - {str(e)}"
                console.print(f"[bold red]处理 {video_file} 时发生异常: {status_msg}[/bold red]")
            finally:
                # ── 恢复配置 & 写回 Excel ──
                update_key('whisper.language', orig_src)
                update_key('target_language', orig_tgt)
                df.at[index, 'Status'] = status_msg
                df.to_excel(SETTINGS_FILE, index=False)
                gc.collect()
                time.sleep(1)
        else:
            print(f"⏭️ 跳过已完成任务: {row['Video File']} - Status: {row['Status']}")

    console.print(Panel(
        "所有字幕任务处理完毕！\n请前往 `batch/output` 查看结果",
        title="[bold green]✅ 批量字幕处理完成",
        expand=False,
    ))


if __name__ == "__main__":
    process_batch_subtitle()
