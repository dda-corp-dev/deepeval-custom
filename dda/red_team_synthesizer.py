import time
import openai
import logging
import transformers

from typing import Optional, Tuple
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from deepeval.red_teaming import RedTeamer
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from openpyxl import Workbook
from datetime import datetime
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from dda.helper import (
    get_titles,
    translate,
    valid_gpt_models,
    default_gpt_model,
    get_attack_prompt,
    get_user_prompt,
    generate_attack_prompts,
    merge_files,
)
from dda.red_team_config import params
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


class TargetGPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )

            if model_version is None:
                model_version = ""

            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
                *self.args,
                **self.kwargs,
            )

        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self._openai_api_key,
            *self.args,
            **self.kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res.content

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            return res.content

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name


class CustomClaudeOpus(DeepEvalBaseLLM):
    def __init__(self):
        self.model = Anthropic(
            api_key=KEY_FILE_HANDLER.fetch_data(KeyValues.ANTHROPIC_API_KEY)
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str):
        client = self.load_model()
        resp = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return resp.content[0].text

    async def a_generate(self, prompt: str):
        return self.generate(prompt)

    def get_model_name(self):
        return "Clause-3 Opus"


class CustomQwen2_3B(DeepEvalBaseLLM):
    def __init__(self):
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        messages = [
            {
                "role": "system",
                "content": "You are an AI Korean assistant designed to fulfill the user's spoken requests.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="cpu",
            max_length=2048,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
        )
        result = pipeline(text)
        return result[0]["generated_text"].split("<|im_start|>assistant\n")[-1]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "QWen-2.5 3B"


class CustomLlama3_1B(DeepEvalBaseLLM):
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        messages = [
            {
                "role": "system",
                "content": "You are an AI Korean assistant designed to fulfill the user's spoken requests.",
            },
            {"role": "user", "content": prompt},
        ]
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=1000,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
        )
        result = pipe(
            messages,
        )
        content = result[0]["generated_text"][-1].get("content", "")
        return content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3.1 1B"


def _get_target_model(name):
    if name == "gpt-3.5-turbo-0125":
        return TargetGPTModel(name)
    elif name == "claude-3-opus-20240229":
        return CustomClaudeOpus()
    elif name == "qwen2-3b":
        return CustomQwen2_3B()
    elif name == "llama3-1b":
        return CustomLlama3_1B()
    return None


def save_results(df, path, params):
    write_wb = Workbook()
    write_ws = write_wb.create_sheet("result")
    write_ws = write_wb.active

    df_list = df.to_dict(orient="records")

    write_ws.append(get_titles())

    # vulnerability_str = ", ".join(str(e) for e in params.get("vulnerabilities"))
    attack_str = ", ".join(
        str(e) for e in list(params.get("attack_enhancements").items())
    )

    for data in df_list:
        vulnerability = str(data.get("Vulnerability", ""))

        input = translate(data.get("Input", ""))
        target = translate(data.get("Target Output", ""))
        reason = translate(data.get("Reason", ""))
        score = data.get("Score", "")

        line = [
            params.get("category"),
            params.get("subcategory"),
            params.get("keywords"),
            params.get("target_purpose"),
            params.get("target_system_prompt"),
            params.get("target_model"),
            params.get("evaluation_model"),
            params.get("synthesizer_model"),
            attack_str,
            vulnerability,
            input,
            target,
            score,
            reason,
        ]
        write_ws.append(line)
    write_wb.save(f"{path}.xlsx")


def main(params):
    """deepeval redteamer를 활용하여 LLM 모델을 공격한 결과데이터를 엑셀파일로 저장하는 함수"""
    start = time.time()

    now = datetime.now().strftime("%y%m%d%H%M")
    file_name = f"breakdown_{params['target_model']}_{params['keyword']}_{now}"
    path = f"./dda/data/{file_name}"

    target_model = _get_target_model(params["target_model"])
    if target_model is None:
        raise ValueError(f"Unsupported target model: {params['target_model']}")

    red_teamer = RedTeamer(
        target_purpose=params["target_purpose"],
        target_system_prompt=params["target_system_prompt"],
        evaluation_model=SchematicGPTModel(params["evaluation_model"]),
        synthesizer_model=SchematicGPTModel(params["synthesizer_model"]),
    )
    results = red_teamer.scan(
        target_model=target_model,
        attacks_per_vulnerability=params["attacks_per_vulnerability"],
        attack_enhancements=params["attack_enhancements"],
        vulnerabilities=params["vulnerabilities"],
    )
    df = red_teamer.vulnerability_scores_breakdown
    print(df)
    save_results(df, path, params)

    end = time.time()
    print(f"{end - start:.2f} sec")


def custom_red_teaming(path, params):
    # Create excel file
    write_wb = Workbook()
    write_ws = write_wb.create_sheet("result")
    write_ws = write_wb.active
    write_ws.append(get_titles())

    # Set up red_teamer for scoring
    red_teamer = RedTeamer(
        target_purpose=params["target_purpose"],
        target_system_prompt=params["target_system_prompt"],
        evaluation_model=SchematicGPTModel("gpt-4o"),
        synthesizer_model=SchematicGPTModel("gpt-4o"),
    )
    v_metrics_map = red_teamer.get_red_teaming_metrics_map()

    for idx, v in enumerate(params["vulnerabilities"]):
        metric: BaseMetric = v_metrics_map.get(v)()

        # Generate attack prompts
        system_prompt = get_attack_prompt(
            "./dda/prompts/red_team_system.txt",
            params["attacks_per_vulnerability"],
        )
        prompt = get_user_prompt(
            "./dda/prompts/red_team_user.txt",
            {
                "category": params["category"],
                "subcategory": params["subcategory"],
                "keywords": params["keywords"],
                "vulnerability": params["vulnerabilities_desc"][idx],
                "target_purpose": params["target_purpose"],
                "target_system_prompt": params["target_system_prompt"],
            },
        )
        results = generate_attack_prompts(
            system_prompt,
            prompt,
            params["target_model"],
        )

        for result in results:
            question = result.get("question")
            answer = result.get("answer")

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
            )
            metric.measure(test_case)

            line = [
                params.get("category"),
                params.get("subcategory"),
                params.get("keywords"),
                params.get("target_purpose"),
                params.get("target_system_prompt"),
                params.get("target_model"),
                params.get("evaluation_model"),
                params.get("synthesizer_model"),
                "PROMPT_INJECTION",
                str(v),
                question,
                answer,
                metric.score,
                translate(metric.reason),
            ]
            print(f"Complete! {line}")

            write_ws.append(line)
    write_wb.save(f"{path}.xlsx")


def custom_main(params):
    now = datetime.now().strftime("%y%m%d%H%M")
    file_name = f"breakdown_{params['target_model']}_{params['category']}_{params['subcategory']}_{params['keywords']}_{now}"
    path = f"./dda/data/{file_name}"
    custom_red_teaming(path, params)


if __name__ == "__main__":
    # 1) DeepEval Redteamer scanning
    main(params)

    # 2) Custom
    custom_main(params)

    # If you want to merge excel data, using merge_files function.
    # merge_files()
