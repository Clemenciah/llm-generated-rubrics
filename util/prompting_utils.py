import pandas as pd 
from prompts.prompt_template import StringFormatter, RoleDescription
from util.sampling import sample_dataframe, get_distinct_random_values, process_dataframe
import time, os, json
from datetime import datetime, timezone


EXPERT_PROMPT = "For each instruction, write a high-quality description about the most capable\
 and suitable agent to answer the instruction. In second person perspective.\
[Instruction]: Make a list of 5 possible effects of deforestation.\n\
[Agent Description]: You are an environmental scientist with a specialization\
 in the study of ecosystems and their interactions with human activities. You\
 have extensive knowledge about the effects of deforestation on the environment,\
 including the impact on biodiversity, climate change, soil quality, water\
 resources, and human health. Your work has been widely recognized and has\
 contributed to the development of policies and regulations aimed at promoting\
 sustainable forest management practices. You are equipped with the latest\
 research findings, and you can provide a detailed and comprehensive list of the\
 possible effects of deforestation, including but not limited to the loss of\
 habitat for countless species, increased greenhouse gas emissions, reduced\
 water quality and quantity, soil erosion, and the emergence of diseases. Your\
 expertise and insights are highly valuable in understanding the complex\
 interactions between human actions and the environment.\n\
 [Instruction]: Identify a descriptive phrase for an eclipse.\
 [Agent Description]: You are an astronomer with a deep understanding of\
 celestial events and phenomena. Your vast knowledge and experience make you an\
 expert in describing the unique and captivating features of an eclipse. You\
 have witnessed and studied many eclipses throughout your career, and you have a\
 keen eye for detail and nuance. Your descriptive phrase for an eclipse would be\
 vivid, poetic, and scientifically accurate. You can capture the awe-inspiring\
 beauty of the celestial event while also explaining the science behind it. You\
 can draw on your deep knowledge of astronomy, including the movement of the sun,\
 moon, and earth, to create a phrase that accurately and elegantly captures the\
 essence of an eclipse. Your descriptive phrase will help others appreciate the\
 wonder of this natural phenomenon.\n\
[Instruction]: Identify the parts of speech in this sentence: \"The dog barked\
 at the postman\".\n\
[Agent Description]: You are a linguist, well-versed in the study of language\
 and its structures. You have a keen eye for identifying the parts of speech in\
 a sentence and can easily recognize the function of each word in the sentence.\
 You are equipped with a good understanding of grammar rules and can\
 differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions,\
 and conjunctions. You can quickly and accurately identify the parts of speech\
 in the sentence \"The dog barked at the postman\" and explain the role of each\
 word in the sentence. Your expertise in language and grammar is highly valuable\
 in analyzing and understanding the nuances of communication.\
[Instruction]: [Description]\n\
[Agent Description]: "

INSTRUCTION_PROMPT = '''{Description}
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{Metric}
Evaluation Steps:
'''

SCORING_PROMPT = """
{Description}
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{Metric}

Evaluation Steps:
{Steps}
{Additional_info}
Sample of Evaluation:
{Sample}
Evaluation of {Metric_name}:\n
"""

TAGGING_PROMPT = """
We queried a student and their teacher to provide assessable metrics for a designed task. Each person should've provided a list of metrics, 
containing "metric name", "description", and "scale" from which I will only give you "metric name" and 
"description" for each metric. I will now provide you with the task description, teacher's set of metrics, 
and student's set of metrics.

Designed Task:
{Description}

Teacher's set of metrics:
{Teacher_metrics}

Student's set of metrics:
{Student_metrics}

I now want you to tag the metrics provided by student. tags are either the name of the metrics 
provided by the teacher, or other aspects. you should give me a list containing dictionaries 
where each dictionary has the keywords of "metric_name" and "tags". for this you should:
* go through each student metric and read its description.
* for each student metric, go through the teacher's metrics.
* if the description or part of the description of the student metric covers the teacher's or most of the Teacher's metric's description, tag the student's metric as the teacher's metric under examine.
* after reviewing all of the teacher metric's for this student metric, if there is any specific aspect that is still covered but not mentioned in the teacher's set, clearly choose a name for the aspect and add it to the tags.
"""
#################Generation functions##########################
def generate_system_prompt(model_name, dataset_name,
                           task_description, use_system, 
                           api_function, response_tool, parsing_tool):
  if use_system:
    system_prompt = generate_expert_prompt(model_name, dataset_name,
                                           task_description, 
                                           api_function, response_tool, parsing_tool)
  else:
    system_prompt = "You are a helpful and knowledgeable AI assistant who provides accurate and detailed responses."
  return system_prompt

def generate_expert_prompt(model_name, dataset_name,
                           task_description, 
                           api, response_tool, parsing_tool):
  # Generate a cache file name based on the api function name
  cache_file_name = f"results/cached_system_prompt/{model_name}_{dataset_name}_cache.json"
  
  # Check if the cache file exists
  if os.path.exists(cache_file_name):
    with open(cache_file_name, 'r') as cache_file:
      cached_result = json.load(cache_file)
      print("Loaded cached result.")
      return cached_result
  
  # initial generation
  task_specific_question = task_description
  prompt = EXPERT_PROMPT.replace("[Description]", task_specific_question)
  messages = [{"role": "user", "content": prompt}]
  output = parsing_tool(api(messages, response_tool, temperature=0))
  print("Expert Identity generated")
  
  # Cache the result
  with open(cache_file_name, 'w') as cache_file:
    json.dump(output, cache_file, indent=4)
    print("Cached the result.")
  return output

def generate_metrics(api_function, parser, set_of_tools, definement_prompt, temperature):
  list_of_results = []
  output = api_function(definement_prompt, set_of_tools, temperature=temperature)
  parsed_results = [parser(output)]
  return parsed_results

def generate_instruction(api_function, parser, set_of_tools, instruction_prompt, temperature):
  count = 0
  while True:
    try:
      output = api_function(instruction_prompt, set_of_tools, temperature=temperature)
      parsed_results = parser(output)
      break
    except:
      count += 1
      if count > 3:
        return "Error occurred in generation."
      print("Error occurred in generation, retrying...")
      pass
  return parsed_results
 

def _log_latency(record: dict, filename="results/diagnostics/latency.jsonl"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def generate_scores(api_function, parser, set_of_tools, scoring_prompt, use_temperature):
    if use_temperature:
        temperatures = [0.3, 0.5, 0.7]
        parsed_results = {"score": [], "reasoning": []}
        for temperature in temperatures:
            t0 = time.perf_counter()
            output = api_function(scoring_prompt, set_of_tools, temperature=temperature)
            t1 = time.perf_counter()
            parsed_output = parser(output)
            t2 = time.perf_counter()

            parsed_results["score"].append(parsed_output["score"])
            parsed_results["reasoning"].append(parsed_output["reasoning"])

            _log_latency({
                "ts": datetime.now(timezone.utc).isoformat(),
                "phase": "score_call",
                "model": getattr(api_function, "__name__", "unknown"),
                "temperature": temperature,
                "prompt_chars": len(scoring_prompt) if isinstance(scoring_prompt, str) else None,
                "api_ms": round((t1 - t0) * 1000, 1),
                "parser_ms": round((t2 - t1) * 1000, 1),
                "total_ms": round((t2 - t0) * 1000, 1),
            })
    else:
        temperature = 0
        t0 = time.perf_counter()
        output = api_function(scoring_prompt, set_of_tools, temperature=temperature)
        t1 = time.perf_counter()
        parsed_results = parser(output)
        t2 = time.perf_counter()

        _log_latency({
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": "score_call",
            "model": getattr(api_function, "__name__", "unknown"),
            "temperature": temperature,
            "prompt_chars": len(scoring_prompt) if isinstance(scoring_prompt, str) else None,
            "api_ms": round((t1 - t0) * 1000, 1),
            "parser_ms": round((t2 - t1) * 1000, 1),
            "total_ms": round((t2 - t0) * 1000, 1),
        })
    return parsed_results

def generate_tags(api_function, parser, formatter, tagging_chain_of_prompt, temperature=0):
  output = api_function(tagging_chain_of_prompt, formatter, temperature=temperature)
  parsed_results = parser(output)
  return parsed_results
  
###################Formatting functions#########################
def format_samples_prompt(data, 
                            use_response, 
                            number_of_shots, 
                            overall_score_column_name, 
                            content_column_name,
                            response_column_name):
  distinct_values = get_distinct_random_values(data, content_column_name, num_values = number_of_shots)
  sampled_dataframe = sample_dataframe(data, content_column_name, distinct_values, overall_score_column_name)
  context_function = lambda x: "[Context] "+ x[content_column_name] + "[/Context]"
  if overall_score_column_name == "overall": # TODO: check for Overall vs overall
    context_function = lambda x: "[Context] "+ x[content_column_name] + "[/Context]\n[Facts] "+ x["fact"]+ "[/Facts]"
  samples_prompt = process_dataframe(sampled_dataframe, context_function, content_column_name, use_response, response_column_name)
  return samples_prompt

def format_samples_prompt_sumpubmed_fixed(full_data,
                                          use_response,
                                          overall_score_column_name,
                                          content_column_name,
                                          response_column_name):
    """
    SumPubMed-only few-shot selector for scoring:
      - uses FULL data (allows leakage)
      - picks EXACTLY 3 examples
      - selection is deterministic (same IDs every run)
    """
    df = full_data.copy()


    id_col = next((c for c in ("id", "sample_id", "doc_id", "row_id", "filename_text") if c in df.columns), None)
    if id_col is None:
        df = df.assign(__stable_id=df.index)
        id_col = "__stable_id"


    df = df.dropna(subset=[content_column_name]).sort_values([id_col, content_column_name], kind="mergesort")

    distinct_values = df[content_column_name].drop_duplicates(keep="first").head(3).tolist()
    sampled_dataframe = sample_dataframe(full_data, content_column_name, distinct_values, overall_score_column_name)
    context_function = lambda x: "[Context] " + x[content_column_name] + " [/Context]"
    samples_prompt = process_dataframe(sampled_dataframe, context_function, content_column_name, use_response, response_column_name)
    return samples_prompt
 
def format_instruction_prompt(system_prompt,
                              task_description,
                              metric):
  
    template = StringFormatter(INSTRUCTION_PROMPT)
    user_prompt = template.apply({
      "Description": task_description,
      "Metric": metric.to_str(),
    })
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
              ]

def format_example_prompt(example,
                          dataset_name,
                          content_column_name,
                          response_column_name
                          ):
  context = example[content_column_name]
  response = example[response_column_name]
  if dataset_name == "USR":
    return f"[Context] {context} [/Context]\n[Fact] {example["fact"]} [/Fact]\n[Response] {response} [/Response]"
  return f"[Context] {context} [/Context]\n[Response] {response} [/Response]"

def format_scoring_prompt(system_prompt,
                          samples,
                          example,
                          task_description,
                          metric
                          ):

    metric_name = metric.name
    metric_str = metric.to_str()
    metric_instruction = metric.instruction
    
    if samples == "":
      additional_info = "\n"
    else:
      additional_info = "\n" + samples
      
    template = StringFormatter(SCORING_PROMPT)
    user_prompt = template.apply({
      "Description": task_description,
      "Metric": metric_str,
      "Steps": metric_instruction,
      "Additional_info": additional_info,
      "Sample": example,
      "Metric_name": metric_name
    })
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
              ]

def format_definition_prompt(system_prompt,
                            task_description,
                            samples_prompt,
                            use_restriction,
                            number_of_metrics
                            ):
    template = StringFormatter("{Description}\n{samples_prompt}\nBased on"\
      "this information define {_metrics_} that we should use to assess the quality of contestants' answers. "\
      "For each metric only include "\
      "these entities: Name, Description, Scale")
    user_prompt = template.apply({
      "Description": task_description,
      "_metrics_": f"{number_of_metrics} metrics" if use_restriction else "a set of metrics",
      "samples_prompt": samples_prompt
    })
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
              ]
    
def format_tagging_prompt(task_description,
                          metrics_list,
                          ground_truth_list):
  template = StringFormatter(TAGGING_PROMPT)
  user_prompt = template.apply({
    "Description": task_description,
    "Teacher_metrics": json.dumps([{"name": metric.name, "description": metric.description} for metric in ground_truth_list], indent = 2),
    "Student_metrics": json.dumps([{"name": metric.name, "description": metric.description} for metric in metrics_list], indent = 2)
  })
  return [
      {
          "role": "user",
          "content": user_prompt
      }
            ]