import random
import numpy as np
import pandas as pd
import os, json, traceback
from datetime import datetime, UTC
from util.prompting_utils import (
    generate_system_prompt,
    format_samples_prompt,
    format_definition_prompt,
    format_instruction_prompt,
    format_scoring_prompt,
    format_example_prompt,
    format_tagging_prompt,
    generate_instruction,
    generate_metrics,
    generate_scores,
    generate_tags, format_samples_prompt_sumpubmed_fixed
)
from util.similarity_tools import merge_similar_metrics
from results.save_and_load import save_results, save_tagging_results
from tqdm import tqdm


def end_to_end_definement(api_function,
                          set_of_tools,
                          run_configuration,
                          dataset,
                          seed=42):
    random.seed(seed)

    # extracting run args
    model_name, use_system, use_sample, use_response, number_of_shots, use_temperature, use_restriction = run_configuration.dump()

    # extract dataset information
    data = dataset.data
    dataset_name = dataset.dataset_name
    task_description = dataset.definement_task_description
    overall_score_column_name = dataset.overall_score_column_name
    content_column_name = dataset.content_column_name
    response_column_name = dataset.response_column_name
    metrics_list = dataset.ground_truth_metrics
    number_of_metrics = len(metrics_list) if dataset_name != "USR" else len(metrics_list) - 1
    # settings_string for saving in the end
    saving_settings_string = run_configuration.to_string()

    # check for expert identity
    system_prompt = generate_system_prompt(model_name,
                                           dataset_name,
                                           task_description, use_system,
                                           api_function, set_of_tools["expert_identity_structure"],
                                           set_of_tools["expert_identity_parser"])
    print("system_prompt:\n", system_prompt)

    # check for using samples
    if use_sample:
        samples_prompt = format_samples_prompt(data,
                                               use_response,
                                               number_of_shots,
                                               overall_score_column_name,
                                               content_column_name,
                                               response_column_name
                                               )

    else:
        samples_prompt = ""
    print("samples_prompt:\n", samples_prompt)
    # finalizing the definition prompt
    definition_chain_of_prompts = format_definition_prompt(system_prompt,
                                                           task_description,
                                                           samples_prompt,
                                                           use_restriction,
                                                           number_of_metrics
                                                           )
    print("definition_chain_of_prompts:\n", definition_chain_of_prompts)

    # defining metrics
    metrics = []
    metrics_dataframe = pd.DataFrame(columns=['metrics'])

    # check for using temperature (when generating the metrics with different temperatures)
    if use_temperature:
        # list of temperatures to use
        temperatures = np.arange(0.5, 1, 0.1, dtype=float)
    else:
        temperatures = [0.7]

    for temperature in temperatures:
        metrics.extend(generate_metrics(api_function, set_of_tools["definement_parser"], set_of_tools["definement"],
                                        definition_chain_of_prompts, temperature))
        print(f"---Generated Metrics with Temperature {temperature}---")


    print("---Saving Metrics---")
    save_results("definition", metrics, dataset_name, saving_settings_string)

    return metrics


def metric_instruction_generation(api_function,
                                  model_name,
                                  set_of_tools,
                                  dataset,
                                  metrics_list,
                                  use_system,
                                  seed=42):
    random.seed(seed)

    # extract dataset information
    dataset_name = dataset.dataset_name
    task_description = dataset.scoring_task_description

    # check for expert identity
    system_prompt = generate_system_prompt(model_name,
                                           dataset_name,
                                           task_description, use_system,
                                           api_function, set_of_tools["expert_identity_structure"],
                                           set_of_tools["expert_identity_parser"])
    print("system_prompt:\n", system_prompt)

    for metric in metrics_list:
        # finalizing the definition prompt
        instruction_chain_of_prompt = format_instruction_prompt(system_prompt,
                                                                task_description,
                                                                metric
                                                                )
        print("instruction_generation_chain_of_prompt:\n", instruction_chain_of_prompt)

        instruction = generate_instruction(api_function,
                                           set_of_tools["instruction_parser"],
                                           set_of_tools["instruction"],
                                           instruction_chain_of_prompt,
                                           temperature=0.7)

        metric.instruction = instruction
        print(f"---Generated Instruction for Metric: {metric.name}---")

    return metrics_list



def process_sample(index, sample, metric,
                   system_prompt, samples_prompt,
                   task_description, api_function,
                   set_of_tools, use_temperature,
                   dataset_name, content_column_name,
                   response_column_name
                   ):
    example = format_example_prompt(sample, dataset_name, content_column_name, response_column_name)
    scoring_chain_of_prompt = format_scoring_prompt(system_prompt,
                                                    samples_prompt,
                                                    example,
                                                    task_description,
                                                    metric
                                                    )
    try:
        evaluation = generate_scores(api_function,
                                     set_of_tools["scoring_parser"],
                                     set_of_tools["scoring"],
                                     scoring_chain_of_prompt,
                                     use_temperature
                                     )
        return index, metric.name, evaluation["score"], evaluation["reasoning"]
    except:
        print(f"Error occurred while evaluating sample: {sample}\n metric: {metric.name}")
        return index, metric.name, None, None


DEBUG_ROOT = os.path.join("results", "scoring_debug")


def _safe_jsonable(obj, maxlen=4000):
    """Make any object JSON-serializable (fallback to repr, truncate long)."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        s = repr(obj)
        if len(s) > maxlen:
            s = s[:maxlen] + f"... [truncated {len(s) - maxlen} chars]"
        return s


def _prompt_len(messages):
    try:
        return sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    except Exception:
        return -1


def _dump_debug(dataset_name, model_name, metric_name, row_index, attempt, prompt, sample, exc):
    """Write a compact debug JSON without ever dumping non-serializable objects."""
    outdir = os.path.join(DEBUG_ROOT, dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "metric": metric_name,
        "row_index": int(row_index),
        "attempt": attempt,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "prompt_char_count": _prompt_len(prompt),
        "prompt": prompt,  # list[{"role","content"}]
        "sample_preview": _safe_jsonable(dict(sample)),  # pandas Series → dict → repr-safe
    }
    fname = f"{metric_name}__row{row_index}__try{attempt}.json"
    with open(os.path.join(outdir, fname), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def end_to_end_scoring(api_function,
                       model_name,
                       set_of_tools,
                       dataset,
                       metrics_list,
                       configuration,
                       overall_assessment_metric=None,
                       seed=42):
    random.seed(seed)
    # extract configuration information
    model_name, use_system, use_sample, number_of_shots, use_temperature = configuration.dump()
    use_response = False

    # extract dataset information
    dataset_name = dataset.dataset_name
    task_description = dataset.scoring_task_description
    test_set = dataset.selected_subset  # 50 random selected samples
    data = dataset.data
    overall_score_column_name = dataset.overall_score_column_name
    content_column_name = dataset.content_column_name
    response_column_name = dataset.response_column_name

    test_less_data = data[~data.apply(tuple, 1).isin(test_set.apply(tuple, 1))]

    # check for expert identity
    system_prompt = generate_system_prompt(model_name,
                                           dataset_name,
                                           task_description, use_system,
                                           api_function, set_of_tools["expert_identity_structure"],
                                           set_of_tools["expert_identity_parser"])
    print("system_prompt:\n", system_prompt)

    if use_sample:
        print(str(dataset_name).lower())
        if str(dataset_name).lower() == "sumpubmed":
            # SumPubMed policy: use FULL data, exactly 3, deterministic (allows leakage)
            samples_prompt = format_samples_prompt_sumpubmed_fixed(
                data,  # FULL dataset here
                use_response,
                overall_score_column_name,
                content_column_name,
                response_column_name
            )
        else:
            # all other datasets keep the existing behavior
            samples_prompt = format_samples_prompt(
                test_less_data,
                use_response,
                number_of_shots,
                overall_score_column_name,
                content_column_name,
                response_column_name
            )


    else:
        samples_prompt = ""
    print("samples_prompt:\n", samples_prompt)

    count = 0
    for metric in metrics_list:
        test_set[f"{metric.name}_score"] = pd.Series([None] * len(test_set), dtype="object")
        test_set[f"{metric.name}_reasoning"] = [""] * len(test_set)
        for index, sample in tqdm(test_set.iterrows(), total=len(test_set), desc=f"Scoring {metric.name}"):
            example = format_example_prompt(sample, dataset_name, content_column_name, response_column_name)
            scoring_chain_of_prompt = format_scoring_prompt(system_prompt,
                                                            samples_prompt,
                                                            example,
                                                            task_description,
                                                            metric
                                                            )
            count = 0
            while True:
                try:
                    evaluation = generate_scores(api_function,
                                                 set_of_tools["scoring_parser"],
                                                 set_of_tools["scoring"],
                                                 scoring_chain_of_prompt,
                                                 use_temperature
                                                 )
                    test_set.loc[index, f"{metric.name}_score"] = evaluation["score"]
                    test_set.loc[index, f"{metric.name}_reasoning"] = evaluation["reasoning"]
                    break
                except Exception as e:
                    count += 1
                    print(f"Error occurred while evaluating sample: {sample}")
                    # dump a debug file for this failure attempt
                    _dump_debug(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        metric_name=metric.name,
                        row_index=index,
                        attempt=count,
                        prompt=scoring_chain_of_prompt,
                        sample=sample,
                        exc=e
                    )
                    if count == 3:
                        test_set.loc[index, f"{metric.name}_score"] = -1
                        test_set.loc[index, f"{metric.name}_reasoning"] = "Error"
                        break
                    continue

    return test_set


def tagger_runner(api_function,
                  metrics_list,
                  dataset_name,
                  task_description,
                  ground_truth_metrics,
                  tag_formatter,
                  tag_parser,
                  saving_settings_string,
                  seed=42):
    random.seed(seed)

    # finalizing the tagging prompt
    tagging_chain_of_prompt = format_tagging_prompt(task_description,
                                                    metrics_list,
                                                    ground_truth_metrics
                                                    )
    print("tagging_chain_of_prompt:\n", tagging_chain_of_prompt)

    tags = generate_tags(api_function,
                         tag_parser,
                         tag_formatter,
                         tagging_chain_of_prompt,
                         temperature=0
                         )

    print("---Saving Tags---")
    save_tagging_results("tag", tags, dataset_name, saving_settings_string)
    return tags
