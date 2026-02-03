from openai import OpenAI
from pydantic import BaseModel
import os, json, logging, re

def gpt_api(messages, response_format, temperature=0.7):
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )
    client.beta.chat.completions.parse

    chat_completion = client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format
    )
    return chat_completion.choices[0].message.parsed
###############################################################################
import os
from mistralai import Mistral

def mistral_api(messages, response_format, temperature=0.7):
    api_key = os.getenv("MISTRALAI_API_KEY")
    model = "open-mixtral-8x22b"
    client = Mistral(api_key=api_key)

    messages[-1]['content'] += response_format()

    chat_response = client.chat.complete(
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 4096,
        stream = False,
    )

    return chat_response.choices[0].message.content
###############################################################################
def qwen_api(messages, response_format, temperature=0.7):
    ## we used DeepInfra, but you can use your preferred platform by changing the code here
    deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
    openai = OpenAI(
        api_key= deepinfra_api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages= messages,
        tools = [response_format],
        temperature = temperature,
        max_tokens = 4096,
        stream= False,
        function_call= {"name":response_format["function"]["name"]},
        tool_choice="required"
    )

    return chat_completion
###############################################################################

log = logging.getLogger(__name__)

def _mk_json_contract_from_response_format(response_format: dict) -> str:
    """
    Build a *minimal* JSON-output contract from the tool's docstring.
    - Allows `score` to be number *or* string.
    - No prose allowed; one JSON object only.
    """
    root_key = "evaluation result"
    inner_score_key = "score"
    inner_reason_key = "reasoning"

    # Try to extract hints from tool description (docstring)
    try:
        desc = response_format.get("function", {}).get("description", "") or ""
        # Look for keys mentioned in the docstring
        if re.search(r'"evaluation result"', desc, re.I):
            root_key = "evaluation result"
        elif re.search(r'"evaluation"', desc, re.I):
            root_key = "evaluation"

        if re.search(r'"reasoning"', desc, re.I): inner_reason_key = "reasoning"
        if re.search(r'"score"', desc, re.I):     inner_score_key = "score"
    except Exception:
        pass

    # Construct an explicit contract
    contract = ( # this can change if it is ineffective
        "Return ONLY a single valid JSON object (no commentary). "
        "Use this structure:\n"
        "{\n"
        f'  "{root_key}": {{\n'
        f'    "{inner_reason_key}": "<string, brief justification>",\n'
        f'    "{inner_score_key}": "<number or string label based on the metric scale>"\n'
        "  }\n"
        "}\n"
        "Rules: Do not include extra keys. If the score is numeric, emit a number (not a string). "
        "If the metric scale is categorical, emit the exact string label."
    )
    return contract, root_key, inner_score_key, inner_reason_key

# wrote this method because the normal one does not work for scoring
def qwen_api_scoring_json_mode(messages, response_format, temperature=0.0):
    """
    SCORING-ONLY for Qwen via DeepInfra using JSON mode.
    """
    ## we used DeepInfra, but you can use your preferred platform by changing the code here
    deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
    openai = OpenAI(api_key=deepinfra_api_key, base_url="https://api.deepinfra.com/v1/openai")

    contract_text, root_key, score_key, reasoning_key = _mk_json_contract_from_response_format(
        response_format or {}
    )

    # Optional
    if os.getenv("LLMRUBRICS_DEBUG"):
        log.info("[qwen-json] contract keys -> root=%r score=%r reasoning=%r",
                 root_key, score_key, reasoning_key)
    json_guard_msg = {"role": "system", "content": contract_text}
    final_messages = [json_guard_msg] + list(messages)

    # Ask DeepInfra to enforce JSON content
    return openai.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=final_messages,
        temperature=temperature,
        max_tokens=4096,
        stream=False,
        response_format={"type": "json_object"},
    )
###############################################################################
def llama_api(messages, response_format, temperature=0.7):
    ## we used DeepInfra, but you can use your preferred platform by changing the code her
    deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
    openai = OpenAI(
        api_key= deepinfra_api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages= messages,
        tools = [response_format],
        temperature = temperature,
        max_tokens = 4096,
        stream= False,
        function_call= {"name":response_format["function"]["name"]},
        tool_choice="required"
    )

    return chat_completion

###############################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
from functools import partial

## This model was not used in this work.
## Vicuna-13B-v1.5 16k
def vicuna_chat_completion(messages, model, formatting_prompt, tokenizer, temperature=0.7, top_p=1):
    conversation = ""
    for message in messages:
        if message['role'] == 'system':
            conversation += f"SYSTEM: {message['content']}\n"
        elif message['role'] == 'user':
            conversation += f"USER: {message['content']}\n"
        else:
            conversation += f"ASSISTANT: {message['content']}\n"
    conversation += formatting_prompt
    conversation += "ASSISTANT: "

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=temperature, top_p = top_p, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("ASSISTANT:")[-1].strip()

def chat_with_accelerate(messages, model, tokenizer, tools, max_new_tokens=4096, temperature=0.7, top_p=1):
    inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=tools,
                return_dict=True,
                return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )


    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response
###############################################################################
def load_api(api_name):
    if 'gpt' in api_name:
        os.environ["OPENAI_MODEL"] = api_name
        return gpt_api
    elif api_name == 'llama':
        return llama_api
    elif api_name == 'qwen':
        return qwen_api
    elif api_name == 'mixtral':
        return mistral_api
    elif api_name == 'vicuna':
        model_name = "lmsys/vicuna-13b-v1.5-16k"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return partial(vicuna_chat_completion, model=model, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown API name: {api_name}")
