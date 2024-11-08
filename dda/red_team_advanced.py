from datetime import datetime
from openpyxl import Workbook
from dda.red_team_config import advanced_params
from dda.helper import (
    get_attack_prompt,
    get_user_prompt,
    generate_attack_prompts,
    generate_response,
)
from openpyxl import load_workbook
from deepeval.metrics.red_teaming_metrics import *
from deepeval.models.gpt_model_schematic import SchematicGPTModel
from deepeval.red_teaming import Vulnerability
from deepeval.test_case import LLMTestCase


def _get_advanced_titles(params):
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
    for metric in params["metrics_str"]:
        titles_total.append(f"{metric}.Score")
        titles_total.append(f"{metric}.Reason")
    return titles_total


def generate_input_prompts(path, params):
    # Create excel file
    write_wb = Workbook()
    write_ws = write_wb.create_sheet("result")
    write_ws = write_wb.active
    write_ws.append(_get_advanced_titles(params))

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
    write_wb.save(f"{path}.xlsx")


def generate_outputs(path, params):
    load_wb = load_workbook(
        path,
        data_only=True,
    )
    load_ws = load_wb["Sheet"]

    for row in load_ws.iter_rows(min_row=2):
        input = row[9].value
        output = generate_response(input, params["target_model"])
        # print(output)
        row[10].value = output
    load_wb.save(path)


def generate_scoring(path, params):
    load_wb = load_workbook(
        path,
        data_only=True,
    )
    load_ws = load_wb["Sheet"]

    harm: BaseMetric = HarmGrader(
        model=SchematicGPTModel("gpt-4o"),
        harm_category=f"{Vulnerability.HATE}\n{Vulnerability.RADICALIZATION}\n{Vulnerability.VIOLENT_CRIME}",
    )
    politic: BaseMetric = PoliticsGrader(
        model=SchematicGPTModel("gpt-4o"),
    )
    religion: BaseMetric = ReligionGrader(
        model=SchematicGPTModel("gpt-4o"),
    )
    bias: BaseMetric = BiasGrader(
        purpose=params["target_purpose"],
        model=SchematicGPTModel("gpt-4o"),
    )

    for row in load_ws.iter_rows(min_row=2):
        input = row[9].value
        output = row[10].value

        test_case = LLMTestCase(
            input=input,
            actual_output=output,
        )

        hr = harm.measure(test_case)
        pr = politic.measure(test_case)
        rr = religion.measure(test_case)
        br = bias.measure(test_case)

        row[11].value = hr.score
        row[12].value = hr.reason

        row[13].value = pr.score
        row[14].value = pr.reason

        row[15].value = rr.score
        row[16].value = rr.reason

        row[17].value = br.score
        row[18].value = br.reason

    load_wb.save(path)


def advanced_main(params):
    now = datetime.now().strftime("%y%m%d%H%M")
    # file_name = (
    #     f"{now}_{params['output_language']}_breakdown_{params['category']}"
    # )
    file_name = "2411081516_KO_breakdown_Violations of human rights_1.3.xlsx"
    path = f"./dda/data/{file_name}"
    generate_scoring(path, params)


if __name__ == "__main__":
    advanced_main(advanced_params)
