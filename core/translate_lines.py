import re
from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness, get_prompt_natural
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
console = Console()


def _norm_text(s: str) -> str:
    """归一化：去所有非字母数字字符并小写，用于宽松比对模型回显的原文"""
    return re.sub(r'[\W_]+', '', str(s).lower())

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)
    line_count = len(lines.split('\n'))
    compact = load_key("compact_translate_output")

    # 兼容精简模式（值=译文字符串）与完整模式（值={"origin","free"} 字典）
    def get_origin(item, i):
        return item['origin'] if isinstance(item, dict) else lines.split('\n')[i - 1]
    def get_free(item):
        return item['free'] if isinstance(item, dict) else item

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name, required_sub_key):
        src_lines = lines.split('\n')
        def valid_def(response_data):
            required_keys = [str(i) for i in range(1, length+1)]
            # 精简模式只要求数字键齐全，不回显原文
            required_sub_keys = [] if compact else [required_sub_key]
            return valid_translate_result(response_data, required_keys, required_sub_keys)
        def origin_aligned(response_data):
            """非紧凑模式：逐行校验模型回显的 origin 与输入行归一化一致。
            防止模型合并/遗漏行导致"行数恰好相等但内容错位"（纯行数校验无法发现）。"""
            if compact:
                return True
            for i in range(1, length + 1):
                item = response_data.get(str(i))
                origin = item.get('origin', '') if isinstance(item, dict) else ''
                if _norm_text(origin) != _norm_text(src_lines[i - 1]):
                    return False
            return True
        for retry in range(3):
            result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_def, log_title=f'translate_{step_name}')
            if len(src_lines) == len(result) and origin_aligned(result):
                return result
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details.[/red]')

    # 单遍自然翻译：忠实直译 + 自然润色一次完成（省 token）
    if not load_key('reflect_translate'):
        prompt = get_prompt_natural(lines, shared_prompt)
        natural_result = retry_translation(prompt, line_count, 'natural', 'free')

        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(natural_result):
            table.add_row(f"[cyan]Origin:  {get_origin(natural_result[key], i + 1)}[/cyan]")
            table.add_row(f"[magenta]Natural: {get_free(natural_result[key])}[/magenta]")
            if i < len(natural_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

        console.print(table)

        translate_result = "\n".join([get_free(natural_result[i]).replace('\n', ' ').strip() for i in natural_result])

        if line_count != len(translate_result.split('\n')):
            console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_natural.json`[/red]'))
            raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

        return translate_result, lines

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, line_count, 'faithfulness', 'direct')

    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, line_count, 'expressiveness', 'free')

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    translate_result = "\n".join([express_result[i]["free"].replace('\n', ' ').strip() for i in express_result])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)