import os, sys
import debugpy
import gc
import shutil
import time
from functools import partial

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from batch.utils.settings_check import check_settings
from core import *
from core._1_ytdlp import find_video_files
from core.utils.config_utils import load_key, update_key
from core.utils.onekeycleanup import cleanup

console = Console()

# ── 常量 ─────────────────────────────────────────────────
SETTINGS_FILE = 'batch/tasks_setting.xlsx'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
INPUT_DIR = 'batch/input'

# ── 调试器初始化（模块级，仅在主动开启时生效）──
# 开启方式：设置环境变量 VIDEOLINGO_DEBUG=1（例如：set VIDEOLINGO_DEBUG=1）
# 关闭方式：不设置该环境变量即可（默认）
# 用法：先运行脚本，然后在 VS Code 按 F5 选择 "Attach to Running VideoLingo"
if os.environ.get("VIDEOLINGO_DEBUG") == "1":
    try:
        debugpy.listen(5678)
        print("[debug] 🟢 调试器已启动，等待 VS Code 附加到 localhost:5678 ...")
        debugpy.wait_for_client()
        print("[debug] 🔵 继续执行（可在 VS Code 中随时设置断点）")
    except Exception as e:
        if "already being" in str(e).lower():
            pass  # 忽略重复 listen 的提示
        else:
            print(f"[debug] ⚠️ 调试器启动失败: {e}")


def record_and_update_config(source_language: str, target_language: str):
    """暂存并切换当前任务的源语言与目标语言配置"""
    original_source_lang = load_key('whisper.language')
    original_target_lang = load_key('target_language')

    if source_language and not pd.isna(source_language):
        update_key('whisper.language', source_language)
    if target_language and not pd.isna(target_language):
        update_key('target_language', target_language)

    return original_source_lang, original_target_lang


def prepare_output_folder(video_file: str = None):
    """清空 output/ 并从 output/audio/<视频名>/ 恢复中间文件"""
    video_name = os.path.splitext(video_file)[0] if video_file else None
    audio_sub_dir = os.path.join(SAVE_DIR, video_name) if video_name else None

    # 1) 先将 output/audio/<视频名>/（如果存在）备份到临时目录
    temp_backup = None
    if audio_sub_dir and os.path.exists(audio_sub_dir):
        temp_backup = os.path.join('batch', 'output', '.temp_audio_backup')
        if os.path.exists(temp_backup):
            shutil.rmtree(temp_backup)
        os.makedirs(temp_backup, exist_ok=True)
        for item in os.listdir(audio_sub_dir):
            src = os.path.join(audio_sub_dir, item)
            dst = os.path.join(temp_backup, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # 2) 清空 output/
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3) 从临时备份恢复到 output/
    if temp_backup and os.path.exists(temp_backup):
        for item in os.listdir(temp_backup):
            src = os.path.join(temp_backup, item)
            dst = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy2(src, dst)
        shutil.rmtree(temp_backup)
        console.print(f"[green]✅ 已从 output/audio/{video_name}/ 恢复中间文件[/green]")


def save_audio_subtitles(video_file: str):
    """将 output/audio/ 下的 .srt 文件保存到 output/audio/<视频名>/"""
    video_name = os.path.splitext(video_file)[0]
    audio_dir = os.path.join(OUTPUT_DIR, 'audio')
    audio_sub_dir = os.path.join(audio_dir, video_name)

    if not os.path.exists(audio_dir):
        return

    os.makedirs(audio_sub_dir, exist_ok=True)

    for filename in os.listdir(audio_dir):
        if filename.endswith('.srt'):
            src = os.path.join(audio_dir, filename)
            dst = os.path.join(audio_sub_dir, filename)
            shutil.copy2(src, dst)


def restore_from_error(video_file: str):
    """从 batch/output/ERROR/<视频名>/ 恢复中间产物到 output/"""
    video_name = os.path.splitext(video_file)[0]
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
    """处理输入文件：本地复制 或 YouTube 下载（仅 dubbing 模式）"""
    if file.startswith('http'):
        _1_ytdlp.download_video_ytdlp(file, resolution=load_key('ytb_resolution'))
        video_file = find_video_files()
    else:
        input_file = os.path.join(INPUT_DIR, file)
        output_file = os.path.join(OUTPUT_DIR, file)
        shutil.copy(input_file, output_file)
        video_file = output_file
    return {'video_file': video_file}


def gen_audio_tasks():
    """生成配音音频任务 + 分割配音块"""
    _8_1_audio_task.gen_audio_task_main()
    _8_2_dub_chunks.gen_dub_chunks()


def process_single_video(video_file: str, is_retry: bool = False):
    """
    对单个视频执行配音管线（假设字幕已存在）。
    返回 (success: bool, error_step: str, error_message: str)
    """
    if not is_retry:
        prepare_output_folder(video_file)

    steps = [
        ("🎥 复制/下载视频", partial(process_input_file, video_file)),
        ("🔊 生成音频任务 + 分割配音块", gen_audio_tasks),
        ("🎵 提取参考音频", _9_refer_audio.extract_refer_audio_main),
        ("🗣️ TTS 生成配音音频", _10_gen_audio.gen_audio),
        ("🔄 合并完整配音音频", _11_merge_audio.merge_full_audio),
        ("🎞️ 配音混入视频", _12_dub_to_vid.merge_video_audio),
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

    # 配音成功后，保存当前字幕文件到 output/audio/<视频名>/ 供后续恢复
    save_audio_subtitles(video_file)

    console.print(Panel("[bold green]🎉 配音生成与混入全部完成！[/bold green]", border_style="green"))
    cleanup(SAVE_DIR)
    return True, "", ""


def process_batch_dubbing():
    """批量配音处理主入口"""
    # ── 前置校验 ──
    if not check_settings():
        raise Exception("配置校验未通过，请检查 batch/tasks_setting.xlsx")

    df = pd.read_excel(SETTINGS_FILE)
    total = len(df)

    for index, row in df.iterrows():
        # 只处理 Dubbing=1 且 DubbingStatus 不为 "Done" 的行
        dubbing = 0 if pd.isna(row['Dubbing']) else int(row['Dubbing'])
        if dubbing != 1:
            print(f"⏭️ 跳过不需配音的任务: {row['Video File']} (Dubbing={dubbing})")
            continue

        dubbing_status = row.get('DubbingStatus')
        if not pd.isna(dubbing_status) and str(dubbing_status) == 'Done':
            print(f"⏭️ 跳过已完成配音任务: {row['Video File']} - DubbingStatus: {dubbing_status}")
            continue

        video_file = row['Video File']
        is_retry = not pd.isna(dubbing_status) and 'Error' in str(dubbing_status)

        # ── 标题面板 ──
        if is_retry:
            console.print(Panel(
                f"重试失败配音任务: {video_file}\n任务 {index + 1}/{total}",
                title="[bold yellow]🔄 重试配音任务",
                expand=False,
            ))
            restore_from_error(video_file)
        else:
            console.print(Panel(
                f"正在配音: {video_file}\n任务 {index + 1}/{total}",
                title="[bold magenta]🎤 当前配音任务",
                expand=False,
            ))

        # ── 语言切换 ──
        source_language = row['Source Language']
        target_language = row['Target Language']
        orig_src, orig_tgt = record_and_update_config(source_language, target_language)

        try:
            success, error_step, error_msg = process_single_video(video_file, is_retry)
            dub_status_msg = "Done" if success else f"Error: {error_step} - {error_msg}"
        except Exception as e:
            dub_status_msg = f"Error: 未捕获异常 - {str(e)}"
            console.print(f"[bold red]配音处理 {video_file} 时发生异常: {dub_status_msg}[/bold red]")
        finally:
            # ── 恢复配置 ──
            update_key('whisper.language', orig_src)
            update_key('target_language', orig_tgt)
            df.at[index, 'DubbingStatus'] = dub_status_msg

            # ── 写回 Excel（多次重试，防止文件被 Excel 占用）──
            for save_attempt in range(5):
                try:
                    df.to_excel(SETTINGS_FILE, index=False)
                    break
                except PermissionError:
                    console.print(f"[yellow]⚠️ 无法写入 {SETTINGS_FILE}（可能被 Excel 打开），"
                                  f"第 {save_attempt + 1}/5 次重试…[/yellow]")
                    time.sleep(2)
                except Exception as e:
                    console.print(f"[red]❌ 写入 {SETTINGS_FILE} 失败: {e}[/red]")
                    break

            gc.collect()
            time.sleep(1)

    console.print(Panel(
        "所有配音任务处理完毕！\n请前往 `batch/output` 查看结果",
        title="[bold green]✅ 批量配音处理完成",
        expand=False,
    ))


if __name__ == "__main__":
    process_batch_dubbing()

