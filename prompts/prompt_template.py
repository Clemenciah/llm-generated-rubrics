from pydantic import BaseModel, Field
from typing import List

class StringFormatter:
    def __init__(self, template):
        """
        Initialize the StringFormatter with a template string.
        :param template: A string with placeholders enclosed in {}.
        """
        self.template = template

    def apply(self, values):
        """
        Replace placeholders in the template with corresponding values from the dictionary.
        :param values: A dictionary containing placeholder keys and their replacement values.
        :return: The formatted string with placeholders replaced.
        """
        try:
            return self.template.format(**values)
        except KeyError as e:
            raise ValueError(f"Missing value for placeholder: {e}") from None
        
def retrieve_set_of_tools(api_name):
    set_of_tools = {}
    if api_name == "vicuna":
        set_of_tools = {
            "simple_response": get_vicuna_void_prompt,
            "definement": get_vicuna_metric_prompt,
            "expert_identity_structure":get_vicuna_expert_identity_structure,
            "scoring":get_vicuna_score_prompt,
            "instruction":get_vicuna_instruction_prompt
        }
    elif "qwen" in api_name:
        set_of_tools = {
            "simple_response": function_to_json(simple_output),
            "definement": function_to_json(define_metrics),
            "expert_identity_structure":function_to_json(role_description_output),
            "scoring":function_to_json(get_metric_score),
            "instruction":function_to_json(set_metric_instruction)
        }
    elif "gpt" in api_name:
        set_of_tools = {
            "simple_response": SimpleResponse,
            "definement": MetricsList,
            "expert_identity_structure":RoleDescription,
            "scoring":MetricScore,
            "instruction":MetricInstruction
        }
    elif "llama" in api_name:
        set_of_tools = {
            "simple_response": function_to_json(simple_output),
            "definement": function_to_json(define_metrics),
            "expert_identity_structure":function_to_json(role_description_output),
            "scoring":function_to_json(get_metric_score),
            "instruction":function_to_json(set_metric_instruction)
        }
    elif "mixtral" in api_name:
        set_of_tools = {
            "simple_response": get_vicuna_void_prompt,
            "definement": get_vicuna_metric_prompt,
            "expert_identity_structure":get_vicuna_expert_identity_structure,
            "scoring":get_vicuna_score_prompt,
            "instruction":get_vicuna_instruction_prompt
        }  
    else:
        set_of_tools = {
            "simple_response": simple_output,
            "definement": define_metrics,
            "expert_identity_structure":role_description_output,
            "scoring":get_metric_score,
            "instruction":set_metric_instruction
        }
       
    return set_of_tools

def function_to_json(func):
    """
    Receives a function and returns its JSON representation.
    :param func: The function to be converted to JSON.
    :return: A dictionary representing the function in JSON format.
    """
    func_name = func.__name__
    func_doc = inspect.getdoc(func)
    func_signature = inspect.signature(func)
    
    parameters = {
        'type': 'object',
        'properties': {},
        'required': []
    }
    
    for name, param in func_signature.parameters.items():
        param_type = str(param.annotation)
        param_description = param.default if param.default is not inspect.Parameter.empty else ''
        parameters['properties'][name] = {
            'title': name,
            'type': param_type,
            'description': param_description
        }
        if param.default is inspect.Parameter.empty:
            parameters['required'].append(name)
    
    return {
        'type': 'function',
        'function': {
            'name': func_name,
            'description': func_doc,
            'parameters': parameters
        }
    }
 
####################GPT-API Prompt Formatting Classes#####################
# Simple response structure
class SimpleResponse(BaseModel):
    response: str = Field(description="your response to the prompt.")
    
# structure for expert identity
class RoleDescription(BaseModel):
    role_description: str = Field(description="description of the role.")

# structure for metric definement
class AbstractMetric(BaseModel):
    name: str = Field(description="name of the metric.")
    description: str = Field(description="description of the metric. What does it measure?") 
    scale: str = Field(description="scale of the metric. What are the possible values or range?")
    
class MetricsList(BaseModel):
    metrics: List[AbstractMetric] = Field(description="List of metrics to be evaluated.")
    
# structure for metric instruction generation
class AbstractStep(BaseModel):
    step_id: int = Field(description="step number.")
    description: str = Field(description="description of the step. What should be done?")
    
class MetricInstruction(BaseModel):
    steps: List[AbstractStep] = Field(description="List of steps to be evaluated.")

# structure for metric scoring  
class MetricScore(BaseModel):
    reasoning: str = Field(description="reasoning behind the score.")
    score: float = Field(description="score based on the provided scale.")

####################Vicuna-API Prompt Formats##################

def get_vicuna_instruction_prompt():
    return'''Always use the following structure:

[
  {
    "step": {
      "step_id": "<step number>",
      "description": "<description of the step>"
    }
  }
]'''

def get_vicuna_metric_prompt():
    return'''Always use the following structure:
[
  {
    "metric": {
      "name": "<metric name>",
      "description": "<description of the metric>",
      "scale": "<scale of the metric>"
    }
  }
]'''

def get_vicuna_score_prompt():
    return'''Provide the reasoning and score using the following structure:

{
  "evaluation result": {
    "reasoning": "<reasoning behind the score>",
    "score": <score based on the provided scale>
  }
}'''

def get_vicuna_expert_identity_structure():
    return'''Provide the reasoning and score using the following structure:
{
    "expert identity": <your response to the prompt.>
}
'''

def get_vicuna_void_prompt():
    return ''''''
####################Other Prompt Formatting functions#######################
from models.metric import Metric
import json
import inspect

def simple_output(response: str) -> str:
    """
    Recieves a string of response in the following structure:
    {
        "response": <your response to the prompt.>
    }

    Args:
        response: The response from the api.

    Returns:
        str: The value of the response.
    """
    response_json = json.loads(response)
    return response_json["response"]
    
def role_description_output(role_description: str) -> str:
    """
    Recieves a string of role description in the following structure:
    {
        "expert identity": <your response to the prompt.>
    }

    Args:
        role_description : _description_
    Returns:
        role_description: The role description.
    """
    role_description_json = json.loads(role_description)
    return role_description_json["expert identity"]
    
def define_metrics(metrics: List[str]) -> List[Metric]:
    """
    Recieves a string of metrics in the following structure:
    [
        {
            "metric": {
                "name": "<metric name>",
                "description": "<description of the metric>",
                "scale": "<scale of the metric>"
            }
        }
    ]
    
    and returns a list of metrics.
    Args:
        metrics: A string of metrics in the above structure.
    Returns:
        A list of Metric objects        
    """
    metric_list = []
    for metric_string in metrics:
        metric_json = json.loads(metric_string)
        metric_list.append(Metric(**metric_json["metric"]))
    return metric_list  

def set_metric_instruction(steps: dict) -> str:
    """
    Receives a dictionary with a "steps" key, where the value is a list of step descriptions.
    The expected structure is:
    {
        "steps": [
            "<step 1 description>",
            "<step 2 description>",
            ...
        ]
    }

    Returns:
        A step-by-step instruction in string format, each step on a new line.
    """
    steps = steps.get("steps", [])
    return "\n".join(f"{idx + 1}. {desc}" for idx, desc in enumerate(steps))

def get_metric_score(evaluation: str) -> dict:
    """
    Recieves a string of the evaluation in the following structure:
    {
        "evaluation result": {
            "reasoning": "<reasoning behind the score>",
            "score": <score based on the provided scale>
        }
    }
    and returns the score and reasoning.
    Args:
        evaluation: The structured evaluation.
    Returns:
        The score and the evaluation reasoning.
    """
    if type(evaluation) == str:
        evaluation = json.loads(evaluation)
    score = evaluation["evaluation result"]["score"]
    reasoning = evaluation["evaluation result"]["reasoning"]
    return {
        "score": score,
        "reasoning": reasoning
    }
    
####################Tagging Prompt Formatting functions######################

# structure for metric tagging
class MetricTags(BaseModel):
    name: str = Field(description="name of the under-examine metric.")
    tags: List[str] = Field(description="a list of tags: either the name of the \
        ground truth metrics covered in this metric, or an external tag.")
class AllMetricsTags(BaseModel):
    metrics_tags: List[MetricTags] = Field(description="a list of metrics and their tags.")

