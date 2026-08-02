import os
import time
import torch
import torchaudio
import yaml
import warnings

from rich import print as rprint

# 参考 Qwen3_ASR 的 Pyannote 离线使用方式：本地模型文件夹 + 修正 config 路径 + 内存波形推理
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
try:
    from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    warnings.filterwarnings("ignore", category=ReproducibilityWarning)
except Exception:
    pass

from pyannote.audio import Pipeline

_sd_pipeline = None


def _fix_pyannote_config(model_path):
    """把 config.yaml 中 embedding/segmentation 路径修正为本地绝对路径（plda 的 $model 占位符 pyannote 会自行解析）"""
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    params = cfg.get("pipeline", {}).get("params", {})
    if "embedding" in params:
        params["embedding"] = os.path.join(model_path, "embedding")
    if "segmentation" in params:
        params["segmentation"] = os.path.join(model_path, "segmentation")

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)


def load_diarization(model_path):
    """加载 Pyannote 离线说话人分离流水线（进程内缓存单例）"""
    global _sd_pipeline
    if _sd_pipeline is not None:
        return _sd_pipeline

    rprint(f"[cyan]🎙️ Loading Pyannote diarization model:[/cyan] {model_path} ...")
    _fix_pyannote_config(model_path)
    _sd_pipeline = Pipeline.from_pretrained(os.path.join(model_path, "config.yaml"))

    if torch.cuda.is_available():
        _sd_pipeline.to(torch.device("cuda"))
        rprint("[green]Pyannote loaded on GPU[/green]")
    return _sd_pipeline


def diarize_audio(audio_file, model_path, num_speakers=0):
    """对整段音频做说话人分离，返回 [[start, end, speaker], ...]；失败时返回 None 不阻断主流程"""
    if not os.path.exists(model_path):
        rprint(f"[yellow]⚠️ Pyannote model not found: {model_path}, skip diarization[/yellow]")
        return None

    t0 = time.time()
    try:
        pipeline = load_diarization(model_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 16k mono 内存波形（与 Qwen3_ASR 的 in-memory 方式一致）
        waveform, sr = torchaudio.load(audio_file)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_in_memory = {"waveform": waveform, "sample_rate": sr}

        sd_kwargs = {}
        if num_speakers and int(num_speakers) > 0:
            sd_kwargs["num_speakers"] = int(num_speakers)
            rprint(f"[cyan]👥 Forced {int(num_speakers)} speakers[/cyan]")

        output = pipeline(audio_in_memory, **sd_kwargs)

        if hasattr(output, "exclusive_speaker_diarization"):
            diarization = output.exclusive_speaker_diarization
        else:
            diarization = output.speaker_diarization

        sd_list = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            sd_list.append([segment.start, segment.end, str(speaker)])

        n_speakers = len(set(str(s[2]) for s in sd_list))
        rprint(f"[green]✅ Diarization done: {n_speakers} speaker(s), took {time.time() - t0:.1f}s[/green]")
        return sd_list

    except Exception as e:
        rprint(f"[red]❌ Pyannote diarization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
