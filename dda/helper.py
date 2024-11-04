import os
import glob
import re
import deepl
import requests
import json

from openpyxl import load_workbook, Workbook
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from anthropic import Anthropic


valid_gpt_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
]

default_gpt_model = "gpt-4o"


def merge_files():
    """root_path 안에 있는 모든 excel파일을 합쳐서 하나의 최종 파일로 생성하는 함수"""

    root_path = f"./dda/data"
    files = glob.glob(os.path.join(root_path, "**", "*.xlsx"), recursive=True)

    final_wb = Workbook()
    final_ws = final_wb.active

    title = get_titles()
    final_ws.append(title)

    for file in files:
        wb = load_workbook(file)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            final_ws.append(row)

    final_wb.save(f"{root_path}/merged.xlsx")


def get_titles():
    titles_total = []
    # Params
    titles_total.append("category")
    titles_total.append("subcategory")
    titles_total.append("keywords")
    titles_total.append("target_purpose")
    titles_total.append("target_system_prompt")
    titles_total.append("target_model")
    titles_total.append("evaluation_model")
    titles_total.append("synthesizer_model")
    titles_total.append("attack")
    # Outputs
    titles_total.append("Vulnerability")
    titles_total.append("Input")
    titles_total.append("Target Output")
    titles_total.append("Score")
    titles_total.append("Reason")
    return titles_total


def _contain_english_string(input):
    # 정규식 패턴: 영어 알파벳이 하나 이상 포함되어 있는지 검사
    eng_regex = re.compile(r"[a-zA-Z]")
    return bool(eng_regex.search(input))


def _translate_deepl(sentence: str, target: str):
    auth_key = KEY_FILE_HANDLER.fetch_data(KeyValues.DEEPL_AUTH_KEY)
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(sentence, target_lang=target)
    return result.text


def translate(input):
    is_english = _contain_english_string(input)
    if is_english:
        return _translate_deepl(input, "KO")
    else:
        return input


def generate_attack_prompts(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
):
    results = []
    try:
        if model_name.find("claude") != -1:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ]
            inputs = get_claude3_answer(
                model_name,
                system_prompt,
                messages,
                0.4,
                8000,
            )
            results = json.loads(inputs).get("result")
        elif model_name.find("gpt") != -1:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            results = get_azure_gpt_answer(
                messages,
                0.4,
                4000,
            )
            results = json.loads(results).get("result")
    except Exception as e:
        print(f"[API ERROR] {e}")
    return results


def get_claude3_answer(
    model: str,
    system: str,
    messages: list,
    temperature: int,
    max_tokens: int,
) -> str:
    try:
        client = Anthropic(
            api_key=KEY_FILE_HANDLER.fetch_data(KeyValues.ANTHROPIC_API_KEY)
        )
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
    except Exception as e:
        print(f"get_claude3_answer error: {e}")
    else:
        return message.content[0].text


def get_azure_gpt_answer(
    messages: list, temperature: float, max_tokens: int
) -> str:
    try:
        AZURE_KEY = KEY_FILE_HANDLER.fetch_data(KeyValues.AZURE_OPENAI_API_KEY)
        AZURE_API_URL = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_ENDPOINT
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_KEY,
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            AZURE_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        print(f"[API REQUEST EXCEPTION] {e}")
    except Exception as e:
        print(f"[API ERROR] {e}")
    else:
        return result["choices"][0]["message"]["content"]


def get_attack_prompt(path: str, num: int):
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read()
    prompt = prompt.replace("#{num}", str(num))
    return prompt


def get_user_prompt(path: str, params: dict):
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read()
    for key in params.keys():
        key_str = f"#{{{key}}}"
        prompt = prompt.replace(key_str, params[key])
    return prompt
