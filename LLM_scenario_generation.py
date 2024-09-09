
from rich.pretty import pprint
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import time
import jwt
import json
from rich.console import Console
from rich.markdown import Markdown
from util import UTIL

console = Console()


def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


if __name__ == "__main__":
    vul = "CVE-2023-32315"
    api_key = "xxx" 
    file_prefix = f"{vul}-{UTIL.current_time}"
    

    generated_scenarios_path = UTIL.scenario_path / "generated_scenarios"

    vul_description_path = UTIL.project_path / "GatheredInfo/vul-description.json"
    with open(vul_description_path, "r", encoding="utf-8") as f:  # *********
        all_vul_description: dict = json.loads(f.read())
    vul_description = all_vul_description[vul]

    example_scenarios_path = UTIL.scenario_path / "auto_probe"

    with open(
        example_scenarios_path / f"{vul}.json", "r", encoding="utf-8"
    ) as f:  # *********
        example_env_data: dict = json.loads(f.read())[0]

    del example_env_data["web_fingerprint_component"]

    llm = ChatOpenAI(
        temperature=0.96,
        model="glm-4",
        openai_api_key=api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    )

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a penetration testing expert with information on various vulnerabilities. Now we are trying to scan a target host with the {CVE_ID} vulnerability using tools such as nmap and whatweb. The {CVE_ID} vulnerability is described as follows: {vul_description}"
            ),
            # MessagesPlaceholder(variable_name="vul_describtion"),
            HumanMessagePromptTemplate.from_template(
                "The following is an example of the scan result: {example}. Since the target hosts (without a restrictive firewall) with {CVE_ID} vulnerability may be scanned differently due to different host configurations (like port, services, os or web fingerprint), please follow the given example, just generate 3 different scan results in JSON format , and explain why these results are reasonable. "
            ),
            # AIMessagePromptTemplate.from_template("Since the target hosts with {CVE_ID} vulnerability may be scanned differently due to different host configurations, please generate three different possible scan results in format like the above example"),
        ]
    )

    prompt_ = prompt.format_prompt(
        CVE_ID=vul, example=example_env_data, vul_description=vul_description
    ).to_messages()

    result = llm(prompt_).content  # .replace("\n", '').strip()
    markdown_result = Markdown(result)

    console.print(markdown_result)
    json_data = []
    for block in result.split("```"):
        if block.startswith("json"):
            data_=block[4:].replace('\n','')
            data = json.loads(data_)
        
            # data["ip"] = example_env_data["ip"]
            json_data.append(data)

    
    
    # pprint(result)
    result_file = open(generated_scenarios_path /f"{file_prefix}-{len(json_data)}.md", "w")
    result_file.write(f"# {vul} \n")
    result_file.write(f"## Describtion \n")
    result_file.write(f"{vul_description} \n")
    result_file.write(f"## Result \n")
    result_file.write(f"{result}")
    
    UTIL.save_json(
        data=json_data, path=generated_scenarios_path / f"{file_prefix}.json"
    )
    result_file.close()

    