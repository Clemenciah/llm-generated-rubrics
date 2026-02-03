import json
import os
import pandas as pd
from typing import List
from models.metric import Metric
import re
from prompts.prompt_template import (
    MetricsList,
    define_metrics,
    RoleDescription,
    role_description_output,
    MetricScore,
    MetricInstruction,
    set_metric_instruction,
    get_metric_score,
    MetricTags,
    AllMetricsTags
)


#################Local tools##################################
def spot_list(api_response: str):
    return api_response[api_response.find('['):api_response.rfind(']') + 1]


def extract_arguments(api_response: str):
    clean_response = api_response[api_response.find('<tool_call>') + 1:api_response.rfind('</tool_call>')]
    response_dict = json.loads(clean_response)
    arguments = response_dict['arguments']
    return arguments


def modify_qwen_string(input_string):
    """
    Finds the second "]" from the right, checks if two characters before it is "}". 
    If false, replaces the "}" before the "]" with "}}".
    """
    try:
        # Find the second ']' from the right
        bracket_index = input_string.rfind(']')

        # Check if the second ']' is found
        if bracket_index == -1:
            return input_string  # Return original string if not found

        modified_string = input_string[:bracket_index] + "}" + input_string[bracket_index:]
        return modified_string
    except IndexError:
        return input_string


#################Definement api parsers#######################
def gpt_api_parser(metrics: MetricsList):
    metrics_list = []
    for abs_metric in metrics.metrics:
        name = abs_metric.name
        description = abs_metric.description
        scale = abs_metric.scale
        metric = Metric(name, description, scale)
        metrics_list.append(metric)
    return metrics_list


def vicuna_api_parser(api_response: str):
    metrics_list = []
    structured_response = json.loads(spot_list(api_response))
    for metric_dict in structured_response:
        metric = Metric(**metric_dict['metric'])
        metrics_list.append(metric)
    return metrics_list


def gen_api_parser(api_response: str):
    list_of_string_metrics = extract_arguments(api_response)
    metrics_list = define_metrics(list_of_string_metrics)
    return metrics_list


def llama_api_parser(api_response):
    api_response = api_response.choices[0].message.tool_calls[0].function.arguments
    metrics_list = []
    metrics = json.loads(api_response)["metrics"]
    if type(metrics) == str:
        metrics = json.loads(metrics)
    for metric_dict in metrics:
        metric = Metric(**metric_dict["metric"])
        metrics_list.append(metric)
    return metrics_list


def qwen_api_parser(api_response):
    api_response = api_response.choices[0].message.tool_calls[0].function.arguments
    metrics_list = []
    metrics = json.loads(api_response)["metrics"]
    if type(metrics) == str:
        metrics = json.loads(metrics)
    for metric_dict in metrics:
        metric = Metric(**metric_dict["metric"])
        metrics_list.append(metric)
    return metrics_list


##################Idnetity api parsers########################
def gpt_identity_parser(api_response: RoleDescription):
    return api_response.role_description


def vicuna_identity_parser(api_response: str):
    try:
        structured_response = json.loads(api_response)
        role_description = structured_response["expert identity"]
        return role_description
    except:
        role_description = api_response
        return role_description


def llama_identity_parser(api_response):
    try:
        try:
            role_description = \
                json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
                    "role_description"]
            return role_description
        except:
            try:
                role_description = \
                    json.loads(api_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
                        "expert identity"]
                return role_description
            except:
                raise ValueError(f"format of response not recognized {api_response}")
    except:
        role_description = api_response["choices"][0]["message"]["content"]
        return role_description


def qwen_identity_parser(api_response):
    try:
        role_description = json.loads(api_response.choices[0].message.tool_calls[0].function.arguments)[
            "role_description"]
        return role_description
    except:
        role_description = api_response.choices[0].message.content
        return role_description


def gen_identity_parser(api_response: str):
    role_description = extract_arguments(api_response)
    role_description = role_description_output(role_description)
    return role_description


###################Instruction Generation parser##############
def gpt_instruction_parser(api_response: MetricInstruction):
    instructions = api_response.steps
    steps = ""
    for step in instructions:
        steps += "\n" + str(step.step_id) + "- " + step.description
    return steps


def vicuna_instruction_parser(api_response: str):
    structured_response = json.loads(api_response)  # this should be a list of dictionaries
    steps = ""
    for step_dict in structured_response:
        step_dict = step_dict["step"]
        steps += "\n" + str(step_dict["step_id"]) + "- " + step_dict["description"]

    return steps


def llama_instruction_parser(api_response):
    try:
        function_call = api_response.choices[0].message.tool_calls[0].function.arguments
        arguments = json.loads(function_call)
        steps = set_metric_instruction(arguments)
        return steps
    except Exception as e:
        print("in exception")
        print(api_response)
        raise e


def qwen_instruction_parser(api_response):
    try:
        function_call = api_response.choices[0].message.tool_calls[0].function.arguments
        arguments = json.loads(function_call)
        steps = set_metric_instruction(arguments)
        return steps
    except Exception as e:
        print("in exception")
        print(api_response)
        raise e


def gen_instruction_parser(api_response: str):
    instructions = api_response
    return instructions


####################Scoring parser############################
def gpt_scoring_parser(api_response: MetricScore):
    return {
        "reasoning": api_response.reasoning,
        "score": api_response.score
    }


def vicuna_scoring_parser(api_response: str):
    structured_response = json.loads(api_response)  # this should a dictionary of the result of an evaluation
    structured_response = structured_response["evaluation result"]
    return {
        "reasoning": structured_response["reasoning"],
        "score": structured_response["score"]
    }

def mixtral_scoring_parser(api_response):
    """
    Unified parser for Mixtral outputs (numeric or string labels).
    Returns:
      {"score": float|str, "reasoning": str}
    """
    import json, re

    resp = _as_dict(api_response)

    def _as_str(x):
        if x is None: return None
        return x if isinstance(x, str) else str(x)

    def _unwrap_eval(d):
        if not isinstance(d, dict): return d
        for k in ("evaluation", "evaluation result", "result", "data"):
            if k in d and isinstance(d[k], dict):
                return d[k]
        return d

    def _is_num_like(s: str) -> bool:
        return bool(re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s.strip()))

    def _balanced_json_from_text(s):
        """Find the first balanced {...} block (handles nested braces)."""
        if not isinstance(s, str): return None
        start = s.find("{")
        if start == -1: return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        return None

    def _extract_tool_args(r):
        if not isinstance(r, dict): return None
        choices = r.get("choices") or []
        if not choices: return None
        ch0 = choices[0]
        msg = ch0.get("message") or {}
        tool_calls = msg.get("tool_calls") or ch0.get("tool_calls") or []
        if tool_calls:
            return tool_calls[0].get("function", {}).get("arguments")
        fc = msg.get("function_call") or ch0.get("function_call")
        if fc and "arguments" in fc:
            return fc["arguments"]
        return None

    def _extract_text(r):
        if isinstance(r, str): return r.strip()
        if not isinstance(r, dict): return None
        choices = r.get("choices") or []
        if choices:
            ch0 = choices[0]
            msg = ch0.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip(): return content.strip()
            text = ch0.get("text")
            if isinstance(text, str) and text.strip(): return text.strip()
        for k in ("content", "text", "response"):
            v = r.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
        return None

    def _from_dict(d, content_text=None):
        d = _unwrap_eval(d)
        raw = (d.get("score") or d.get("label") or d.get("rating")
               or d.get("value") or d.get("verdict"))
        # nested/double-encoded score
        if isinstance(raw, str) and raw.strip().startswith("{") and raw.strip().endswith("}"):
            try: raw = json.loads(raw)
            except Exception: pass
        if isinstance(raw, dict):
            # e.g., {"score": {"label":"Pass"}}
            return _from_dict(raw, content_text=content_text)

        reasoning = _as_str(d.get("reasoning") or d.get("explanation") or d.get("rationale") or content_text or "")

        # numeric
        if isinstance(raw, (int, float)) or (isinstance(raw, str) and _is_num_like(raw)):
            return {"score": float(raw), "reasoning": reasoning}
        if isinstance(raw, bool):
            return {"score": "true" if raw else "false", "reasoning": reasoning}
        # string label (Pass/Fail/High/Medium/Low/etc.)
        if isinstance(raw, str):
            return {"score": raw.strip(), "reasoning": reasoning}

        # No explicit score → try text-like fields
        txt = d.get("content") or d.get("text") or d.get("response")
        if isinstance(txt, str):
            return _from_text(txt, content_text=txt)

        raise ValueError("Missing or unparseable 'score' in evaluation object.")

    def _from_text(s, content_text=None):
        s = _as_str(s) or ""


        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
        if m:
            try:
                return _from_dict(json.loads(m.group(1)), content_text=s)
            except Exception:
                pass


        js = _balanced_json_from_text(s)
        if js:
            try:
                return _from_dict(json.loads(js), content_text=s)
            except Exception:
                # continue to regex fallbacks if JSON is slightly invalid
                pass

        # Key:Value anywhere in text → accept strings or numbers
        m = re.search(r'(?i)\b(score|label|rating|value|verdict)\b\s*[:=]\s*("?)([^"\n}]+)\2', s)
        if m:
            val = m.group(3).strip()
            if _is_num_like(val):
                return {"score": float(val), "reasoning": content_text or s}
            return {"score": val, "reasoning": content_text or s}


        try:
            return _coerce_eval(s, content_text=content_text or s)
        except Exception:
            raise ValueError(f"Could not parse a score/label from text: {s[:200]}")

    # -------- parse path --------
    args = _extract_tool_args(resp)
    if args is not None:
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: pass
        ev = args
        if isinstance(args, dict):
            ev = args.get("evaluation") or args.get("evaluation result") or args
            if isinstance(ev, str) and ev.strip().startswith("{") and ev.strip().endswith("}"):
                try: ev = json.loads(ev)
                except Exception: pass
        return _from_dict(ev) if isinstance(ev, dict) else _from_text(ev, content_text=_as_str(ev))


    content = _extract_text(resp)
    if content is None:
        sample = _as_str(resp)
        raise ValueError(f"Mixtral parser: no content and no tool/function arguments. Raw: {sample[:500] if sample else sample}")

    return _from_text(content, content_text=content)

def _as_dict(resp):
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    for attr in ("to_dict", "dict"):
        if hasattr(resp, attr):
            try:
                return getattr(resp, attr)()
            except Exception:
                pass
    return resp  # leave as-is


def _coerce_eval(evaluation, content_text=None):
    """
    Make sure we end up with {'score': <float>, 'reasoning': <str>}.
    Handles dicts, JSON strings, 'score: X' text, or plain numbers.
    """
    import json, re


    if isinstance(evaluation, dict):
        if "score" in evaluation and "reasoning" in evaluation:
            return evaluation


    if isinstance(evaluation, str):
        s = evaluation.strip()


        try:
            ev2 = json.loads(s)
            if isinstance(ev2, dict) and "score" in ev2 and "reasoning" in ev2:
                return ev2
        except Exception:
            pass


        m = re.search(r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', s, re.I)
        if m:
            score = float(m.group(1))
            mr = re.search(r'"?reasoning"?\s*[:=]\s*("?)(.+?)\1\s*$', s, re.I | re.S)
            reasoning = mr.group(2).strip() if mr else (content_text or s)
            return {"score": score, "reasoning": reasoning}


        if re.fullmatch(r'-?\d+(?:\.\d+)?', s):
            return {"score": float(s), "reasoning": content_text or ""}


    raise ValueError("Could not coerce evaluation into {score, reasoning}")


def llama_scoring_parser(api_response):
    import json, re, traceback

    def pv(x, n=300):
        try:
            s = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        except Exception:
            s = repr(x)
        s = s.replace("\n", "\\n")
        return s if len(s) <= n else f"{s[:n]}... (+{len(s)-n} chars)"

    api_response = _as_dict(api_response)
    print("[llama_parser] START type=", type(api_response).__name__)


    try:
        choices = api_response.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        print("[llama_parser] choices n=", len(choices), "finish_reason=", finish_reason)

        if choices:
            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []
            print("[llama_parser] tool_calls present=", bool(tool_calls), "len=", len(tool_calls))

            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments")
                print("[llama_parser] tool.args type=", type(args).__name__, "preview=", pv(args))


                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                        print("[llama_parser] tool.args json.loads OK")
                    except Exception as e:
                        print("[llama_parser] tool.args json.loads FAILED:", e)

                evaluation = args if args is not None else None
                if isinstance(evaluation, dict):
                    before_keys = list(evaluation.keys())
                    evaluation = evaluation.get("evaluation") or evaluation.get("evaluation result") or evaluation
                    print("[llama_parser] unwrap eval key; before_keys=", before_keys,
                          "after_type=", type(evaluation).__name__)

                else:
                    print("[llama_parser] eval from args STRING preview=", pv(evaluation))


                if isinstance(evaluation, str):
                    s = evaluation.strip()
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            evaluation = json.loads(s)
                            print("[llama_parser] inner eval json.loads OK")
                        except Exception as e:
                            print("[llama_parser] inner eval json.loads FAILED:", e)


                if isinstance(evaluation, dict) and "evaluation result" in evaluation and isinstance(evaluation["evaluation result"], dict):
                    evaluation = evaluation["evaluation result"]
                    print("[llama_parser] unwrapped 'evaluation result' keys=", list(evaluation.keys()))


                if isinstance(evaluation, dict):
                    raw = (evaluation.get("score") or evaluation.get("label")
                           or evaluation.get("value") or evaluation.get("verdict"))
                    reasoning = (evaluation.get("reasoning") or evaluation.get("explanation")
                                 or evaluation.get("rationale") or "")
                    print("[llama_parser] dict eval; raw_type=", type(raw).__name__,
                          "raw_preview=", pv(raw), "reason_len=", len(str(reasoning)))
                    if raw is not None:
                        if isinstance(raw, (int, float)) or (isinstance(raw, str) and re.fullmatch(r'[-+]?\d+(?:\.\d+)?', raw.strip())):
                            print("[llama_parser] RETURN numeric score=", raw)
                            return {"score": float(raw), "reasoning": str(reasoning)}
                        print("[llama_parser] RETURN string score=", pv(raw))
                        return {"score": str(raw).strip(), "reasoning": str(reasoning)}


                coerced = _coerce_eval(evaluation, content_text=None)
                print("[llama_parser] RETURN coerced(tool) score=", coerced["score"],
                      "reason_len=", len(str(coerced.get("reasoning",""))))
                return {"score": coerced["score"], "reasoning": coerced["reasoning"]}
    except Exception as e:
        print("[llama_parser] EXC tool_path:", e, "\n", traceback.format_exc())


    try:
        content = (api_response.get("choices", [{}])[0].get("message", {}) or {}).get("content")
        print("[llama_parser] content type=", type(content).__name__,
              "len=", (len(content) if isinstance(content, str) else None),
              "preview=", pv(content))

        if not isinstance(content, str) or not content.strip():
            raise ValueError("No content to parse in llama_scoring_parser.")

        # Find first JSON object in text (works for <function=...>{...}</function> wrappers)
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            obj_txt = m.group(0)
            print("[llama_parser] JSON match len=", len(obj_txt))
            obj = json.loads(obj_txt)
            evaluation = obj.get("evaluation") or obj.get("evaluation result") or obj
            print("[llama_parser] content obj keys=", (list(obj.keys()) if isinstance(obj, dict) else None))
        else:
            print("[llama_parser] no JSON in content; using raw content")
            evaluation = content  # prose only

        # If evaluation is a JSON string (double-encoded), unwrap once
        if isinstance(evaluation, str):
            s = evaluation.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    evaluation = json.loads(s)
                    print("[llama_parser] inner eval(json string) json.loads OK")
                except Exception as e:
                    print("[llama_parser] inner eval(json string) json.loads FAILED:", e)

        if isinstance(evaluation, dict):
            if "evaluation result" in evaluation and isinstance(evaluation["evaluation result"], dict):
                evaluation = evaluation["evaluation result"]
                print("[llama_parser] unwrapped eval_result (content); keys=", list(evaluation.keys()))
            raw = (evaluation.get("score") or evaluation.get("label")
                   or evaluation.get("value") or evaluation.get("verdict"))
            reasoning = (evaluation.get("reasoning") or evaluation.get("explanation")
                         or evaluation.get("rationale") or content or "")
            print("[llama_parser] dict eval(content); raw_type=", type(raw).__name__,
                  "raw_preview=", pv(raw), "reason_len=", len(str(reasoning)))
            if raw is not None:
                if isinstance(raw, (int, float)) or (isinstance(raw, str) and re.fullmatch(r'[-+]?\d+(?:\.\d+)?', raw.strip())):
                    print("[llama_parser] RETURN numeric(content) score=", raw)
                    return {"score": float(raw), "reasoning": str(reasoning)}
                print("[llama_parser] RETURN string(content) score=", pv(raw))
                return {"score": str(raw).strip(), "reasoning": str(reasoning)}


        coerced = _coerce_eval(evaluation, content_text=content)
        print("[llama_parser] RETURN coerced(content) score=", coerced["score"],
              "reason_len=", len(str(coerced.get("reasoning",""))))
        return {"score": coerced["score"], "reasoning": coerced["reasoning"]}

    except Exception as e:
        print("[llama_parser] EXC content_path:", e, "\n", traceback.format_exc())
        raise



def qwen_scoring_parser(api_response):
    """
    Parse DeepInfra JSON-mode response for Qwen.
    Accepts any of:
      {"evaluation result": {"score": ..., "reasoning": ...}}
      {"evaluation": {"score": ..., "reasoning": ...}}
      {"score": ..., "reasoning": ...}
    """
    api_response = _as_dict(api_response)
    content = api_response["choices"][0]["message"].get("content", "")
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        obj = json.loads(m.group(0)) if m else {}

    evaluation = obj.get("evaluation result") or obj.get("evaluation") or obj
    evaluation = _coerce_eval(evaluation, content_text=content)
    return {"score": evaluation["score"], "reasoning": evaluation["reasoning"]}


def gen_scoring_parser(api_response: str):
    evaluation = api_response
    return evaluation


###################Parser classes#############################
class DefinitionParser:

    def __init__(self, api_name):
        self.parser = self.get_parser(api_name)

    def get_parser(self, api_name):
        if api_name == 'vicuna':
            return vicuna_api_parser
        elif api_name == 'mixtral':
            return vicuna_api_parser
        elif 'gpt' in api_name:
            return gpt_api_parser
        elif 'llama' in api_name:
            return llama_api_parser
        elif 'qwen' in api_name:
            return qwen_api_parser
        else:
            return gen_api_parser


class IdentityParser:

    def __init__(self, api_name):
        self.parser = self.get_parser(api_name)

    def get_parser(self, api_name):
        if api_name == 'vicuna':
            return vicuna_identity_parser
        elif 'gpt' in api_name:
            return gpt_identity_parser
        elif 'llama' in api_name:
            return llama_identity_parser
        elif api_name == 'mixtral':
            return vicuna_identity_parser
        elif api_name == 'qwen':
            return qwen_identity_parser
        else:
            return gen_identity_parser


class InstructionParser:
    def __init__(self, api_name):
        self.parser = self.get_parser(api_name)

    def get_parser(self, api_name):
        if api_name == 'vicuna':
            return vicuna_instruction_parser
        elif 'gpt' in api_name:
            return gpt_instruction_parser
        elif 'llama' in api_name:
            return llama_instruction_parser
        elif api_name == 'mixtral':
            return vicuna_instruction_parser
        elif api_name == 'qwen':
            return qwen_instruction_parser
        else:
            return gen_instruction_parser


class ScoringParser:
    def __init__(self, api_name):
        self.parser = self.get_parser(api_name)

    def get_parser(self, api_name):
        if api_name == 'vicuna':
            return vicuna_scoring_parser
        elif 'gpt' in api_name:
            return gpt_scoring_parser
        elif 'llama' in api_name:
            return llama_scoring_parser
        elif api_name == 'mixtral':
            return mixtral_scoring_parser
        elif api_name == 'qwen':
            return qwen_scoring_parser
        else:
            return gen_scoring_parser


def extract_json_from_content(content):
    """Extracts JSON data from a text file."""
    try:
        # Assuming the JSON data is enclosed in curly braces {}
        start_index = content.find('```json')
        end_index = content.rfind('```')
        if start_index != -1 and end_index != -1:
            json_string = content[start_index + 8: end_index]
            return json.loads(json_string)
        else:
            print(f"No valid JSON found in {content}")
            return content

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {content}: {e}")
        return content


def extract_json_from_list_of_contents(list_of_contents):
    return [extract_json_from_content(content) for content in list_of_contents]


def create_dataframe(criteria_data):
    rows = []

    if isinstance(criteria_data, dict):
        if len(criteria_data) == 1 and isinstance(next(iter(criteria_data.values())), list):
            items = next(iter(criteria_data.values()))
            for item in items:
                rows.append({
                    'name': item.get('Name', ''),
                    'description': item.get('Description', ''),
                    'scale': item.get('Scale', '')
                })
        else:
            for item in criteria_data.values():
                rows.append({
                    'name': item.get('Name', ''),
                    'description': item.get('Description', ''),
                    'scale': item.get('Scale', '')
                })


    elif isinstance(criteria_data, list):
        for item in criteria_data:
            rows.append({
                'name': item.get('Name', ''),
                'description': item.get('Description', ''),
                'scale': item.get('Scale', '')
            })
    return rows


def parse_metrics(metrics_dataframe):
    metrics_dataframe['metrics'] = metrics_dataframe["results"].apply(extract_json_from_list_of_contents)
    rows = []
    for content in metrics_dataframe["metrics"]:
        for metrics in content:
            rows += create_dataframe(metrics)

    metric_description_df = pd.DataFrame(rows, columns=['name', 'description', 'scale'])
    return metric_description_df


##################Tag Parser#################################

def gpt_tag_parser(metricstags: AllMetricsTags):
    metric_tags = []
    for metric_tag in metricstags.metrics_tags:
        name = metric_tag.name
        tags = metric_tag.tags
        metric_tags.append({"name": name, "tags": tags})
    return metric_tags
