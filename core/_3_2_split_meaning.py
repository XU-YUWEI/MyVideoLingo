import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING
console = Console()

def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]

# 本地启发式分句用到的连接词
_CONJUNCTION_WORDS = {
    'and', 'or', 'but', 'so', 'because', 'while', 'when', 'if', 'that', 'which',
    'who', 'where', 'then', 'yet', 'though', 'although', 'since', 'until',
    'before', 'after', 'with', 'without'
}

def heuristic_split_sentence(sentence, num_parts, word_limit):
    """本地启发式分句：在标点/连接词附近切分，保证每段 ≤ word_limit 词。
    切分失败（无法满足要求）返回 None，由调用方回退到 LLM。"""
    tokens = sentence.split()
    total = len(tokens)
    if num_parts <= 1 or total > num_parts * word_limit:
        return None
    # 收集自然切分点：逗号/分号/括号之后，或连接词之前
    boundaries = set()
    for i, tok in enumerate(tokens):
        if tok.endswith((',', ';', '(', ')')):
            boundaries.add(i + 1)
        if i > 0 and tok.strip('()').lower() in _CONJUNCTION_WORDS:
            boundaries.add(i)
    boundaries = sorted(b for b in boundaries if 0 < b < total)
    target = total / num_parts
    parts = []
    start = 0
    for k in range(num_parts - 1):
        ideal = start + target
        lo = start + 1
        hi = min(start + word_limit, total - (num_parts - k - 1))  # 为剩余部分留空间
        if lo > hi:
            return None
        cands = [b for b in boundaries if lo <= b <= hi]
        if cands:
            cut = min(cands, key=lambda b: abs(b - ideal))
        else:
            cut = max(lo, min(int(round(ideal)), hi))
        parts.append(' '.join(tokens[start:cut]))
        start = cut
    parts.append(' '.join(tokens[start:total]))
    if any(len(p.split()) > word_limit for p in parts):
        return None
    return parts

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    # 启发式优先：能本地切分就不调 LLM，省 token（split_meaning_heuristic_first=True 时启用）
    if load_key("split_meaning_heuristic_first"):
        heuristic_parts = heuristic_split_sentence(sentence, num_parts, word_limit)
        if heuristic_parts is not None:
            best_split = '\n'.join(heuristic_parts)
            if index != -1:
                console.print(f'[green]✅ Sentence {index} has been successfully split (heuristic)[/green]')
            table = Table(title="")
            table.add_column("Type", style="cyan")
            table.add_column("Sentence")
            table.add_row("Original", sentence, style="yellow")
            table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
            console.print(table)
            return best_split

    split_prompt = get_split_prompt(sentence, num_parts, word_limit)
    def valid_split(response_data):
        if "split" not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        if "[br]" not in response_data["split"]:
            return {"status": "error", "message": "Split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}
    
    # 分句用独立模型（默认 qwen3-8k，指令遵循好、直接输出 JSON；qwen3-4k 易输出超长分析被截断）
    response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json', valid_def=valid_split, log_title='split_by_meaning', model=load_key("split_model"))
    best_split = response_data["split"]
    split_points = find_split_positions(sentence, best_split)
    # split the sentence based on the split points
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = best_split.split('\n')
            last_part = parts[-1]
            parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
            best_split = '\n'.join(parts)
    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)
    
    return best_split

def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            # print("Tokenization result:", tokens)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # read input sentences
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=load_key("max_split_length"), max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()