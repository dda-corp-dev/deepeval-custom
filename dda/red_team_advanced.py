from datetime import datetime
from openpyxl import Workbook
from dda.red_team_config import advanced_params
from dda.helper import (
    get_attack_prompt,
    get_user_prompt,
    generate_attack_prompts,
    generate_response,
    translate,
    merge_files,
)
from openpyxl import load_workbook
from deepeval.metrics.red_teaming_metrics import *
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.red_teaming import Vulnerability
from deepeval.test_case import LLMTestCase


def _get_advanced_titles(params: dict):
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
    titles_total.append("Input")
    titles_total.append("Target Output")
    for metric in params["metrics"]:
        titles_total.append(f"{metric}.Score")
        titles_total.append(f"{metric}.Reason")
    return titles_total


def _save_input_results(path: str, params: dict, results: list):
    # Create excel file
    write_wb = Workbook()
    write_ws = write_wb.create_sheet("result")
    write_ws = write_wb.active
    write_ws.append(_get_advanced_titles(params))

    for result in results:
        question = result.get("question")
        line = [
            params.get("category"),
            params.get("subcategory"),
            params.get("keywords"),
            params.get("target_purpose"),
            params.get("target_system_prompt"),
            params.get("target_model"),
            params.get("evaluation_model"),
            params.get("synthesizer_model"),
            "basic adversarial prompts, prompt probing, prompt injection, jailbreak (including linear, tree, and crescendo types)",
            question,
        ]
        print(line)
        write_ws.append(line)
    write_wb.save(path)


def generate_input_prompts(path: str, params: dict):
    system_prompt = get_attack_prompt(
        params,
        "./dda/prompts/red_team_system_advanced.txt",
        params["nums"],
    )
    prompt = get_user_prompt(
        "./dda/prompts/red_team_user_advanced.txt",
        {
            "vulnerability": params["vulnerability"],
            "category": params["category"],
            "subcategory": params["subcategory"],
            "keywords": params["keywords"],
            "target_purpose": params["target_purpose"],
            "target_system_prompt": params["target_system_prompt"],
        },
    )

    results = generate_attack_prompts(
        system_prompt,
        prompt,
        params["synthesizer_model"],
    )

    if len(results) >= 0:
        _save_input_results(path, params, results)


def generate_outputs(path: str, params: dict):
    load_wb = load_workbook(
        path,
        data_only=True,
    )
    load_ws = load_wb["Sheet"]

    for idx, row in enumerate(load_ws.iter_rows(min_row=2)):
        input = row[9].value
        if params["output_language"] == "EN":
            input += " Please respond within 500 characters."
        else:
            input += " 500자 이내로 답변해줘."
        row[10].value = generate_response(input, params["target_model"])
        print(f"{idx} {input} completed!")
    load_wb.save(path)    


def generate_scoring(path: str, params: dict):
    load_wb = load_workbook(
        path,
        data_only=True,
    )
    load_ws = load_wb["Sheet"]

    model = SchematicGPTModel("gpt-4o")
    graders = {
        "harm": HarmGrader(model=model, harm_category=f"{Vulnerability.HATE}\n{Vulnerability.RADICALIZATION}\n{Vulnerability.VIOLENT_CRIME}"),
        "politic": PoliticsGrader(model=SchematicGPTModel("gpt-4o")),
        "religion": ReligionGrader(model=SchematicGPTModel("gpt-4o")),
        "bias": BiasGrader(model=SchematicGPTModel("gpt-4o"), purpose=params["target_purpose"])
    }

    def translate_if_needed(text: str):
        return translate(text) if params["output_language"] == "KO" else text

    try:
        for idx, row in enumerate(load_ws.iter_rows(min_row=2)):
            input = row[9].value
            output = row[10].value

            if output:
                test_case = LLMTestCase(
                    input=input,
                    actual_output=output,
                )

                if any(row[i].value is None for i in range(12, 19, 2)):
                    for i, (key, grader) in zip(range(11, 19, 2), graders.items()):
                        grader.measure(test_case)
                        row[i].value = grader.score
                        row[i + 1].value = translate_if_needed(grader.reason)

            print(f"{idx} {input} Completed.")
    except Exception as e:
        print(f"Generate_scoring error occured. msg: {e}")
    finally:
        load_wb.save(path)


def advanced_main(params):
    now = datetime.now().strftime("%y%m%d%H%M")
    file_name = (
        f"{now}_{params['output_language']}_breakdown_{params['category']}"
    )
    # file_name = "2411111448_KO_breakdown_Violations of human rights"
    path = f"./dda/data/{file_name}.xlsx"

    generate_input_prompts(path, params)
    generate_outputs(path, params)
    generate_scoring(path, params)


if __name__ == "__main__":
    advanced_main(advanced_params)
    # merge_files("./dda/data/ko", _get_advanced_titles(advanced_params))
