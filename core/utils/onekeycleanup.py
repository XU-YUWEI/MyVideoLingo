import os
import glob
import hashlib
from core._1_ytdlp import find_video_files
import shutil

def get_video_history_name(video_name):
    """生成用于 batch/output、batch/output/ERROR 下的文件夹名。
    超长标题（>48 字符）截断到 40 字符并追加 8 位哈希后缀，
    避免 Windows 260 字符路径限制（MAX_PATH）导致移动/恢复失败。"""
    name = sanitize_filename(video_name)
    if len(name) <= 48:
        return name
    suffix = hashlib.md5(video_name.encode('utf-8')).hexdigest()[:8]
    return name[:40] + '_' + suffix

def cleanup(history_dir="history"):
    # Get video file name
    video_file = find_video_files()
    video_name = video_file.split("/")[1]
    video_name = os.path.splitext(video_name)[0]

    # Create required folders
    os.makedirs(history_dir, exist_ok=True)
    video_history_dir = os.path.join(history_dir, get_video_history_name(video_name))
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)

    # Move non-log files
    for file in glob.glob("output/*"):
        if not file.endswith(('log', 'gpt_log')):
            move_file(file, video_history_dir)

    # Move log files
    for file in glob.glob("output/log/*"):
        move_file(file, log_dir)

    # Move gpt_log files
    for file in glob.glob("output/gpt_log/*"):
        move_file(file, gpt_log_dir)

    # Delete empty output directories
    try:
        os.rmdir("output/log")
        os.rmdir("output/gpt_log")
        os.rmdir("output")
    except OSError:
        pass  # Ignore errors when deleting directories

def move_file(src, dst):
    try:
        # Get the source file name
        src_filename = os.path.basename(src)
        # Use os.path.join to ensure correct path and include file name
        dst = os.path.join(dst, sanitize_filename(src_filename))
        
        if os.path.exists(dst):
            if os.path.isdir(dst):
                # If destination is a folder, try to delete its contents
                shutil.rmtree(dst, ignore_errors=True)
            else:
                # If destination is a file, try to delete it
                os.remove(dst)
        
        shutil.move(src, dst, copy_function=shutil.copy2)
        print(f"✅ Moved: {src} -> {dst}")
    except PermissionError:
        print(f"⚠️ Permission error: Cannot delete {dst}, attempting to overwrite")
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            print(f"✅ Copied and deleted source file: {src} -> {dst}")
        except Exception as e:
            print(f"❌ Move failed: {src} -> {dst}")
            print(f"Error message: {str(e)}")
    except Exception as e:
        print(f"❌ Move failed: {src} -> {dst}")
        print(f"Error message: {str(e)}")

def sanitize_filename(filename):
    # Remove or replace disallowed characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

if __name__ == "__main__":
    cleanup()