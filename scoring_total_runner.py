import subprocess
import os
import glob

# you can also add settings like this
# files = [
#     'model_name=qwen-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
#     'gpt-4o_ground_truth', 'gpt-4o_ground_truth-use_sample=True',
#     'gpt-4o-mini_ground_truth', 'gpt-4o-mini_ground_truth-use_sample=True',
#     'llama_ground_truth', 'llama_ground_truth-use_sample=True',
#     'mixtral_ground_truth', 'mixtral_ground_truth-use_sample=True',
#     'qwen_ground_truth', 'qwen_ground_truth-use_sample=True'
# ]

files = ['model_name=gpt-4o-mini-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json']


#dataset_names = ["HelpSteer2", "SummEval", "SumPubMed", "USR"]
dataset_names = ["USR"]
results_dir = os.path.join("results", "definition")

for json_file in files:
    if "llama" in json_file:
        api = "llama"
    elif "gpt-4o-mini" in json_file:
        api = "gpt-4o-mini"
    elif "gpt-4o" in json_file:
        api = "gpt-4o"
    elif "qwen" in json_file:
        api = "qwen"
    elif "mixtral" in json_file:
        api = "mixtral"
    else:
        raise ValueError(f"API not found in {json_file}")
    if "use_system=True" in json_file:
        use_system = True
    else:
        use_system = False
    if "use_sample=True" in json_file:
        use_sample = True
    else:
        use_sample = False
    if ".json" not in json_file:
        if "use_sample=True" in json_file:
            json_file = "ground_truth_use_sample"
        else:
            json_file = "ground_truth"
    for dataset in dataset_names:
        #use_sample = True for when we want to score the definitions that do not have samples
        data = dataset
        subprocess_list = ["poetry", "run", "python", os.path.join(".", "scripts", "scoring.py"), "--data", dataset,
                           "--api", api, "--setting_name", json_file]
        if use_system:
            subprocess_list.append("--use_system")
        if use_sample:
            subprocess_list.append("--use_sample")
        print(subprocess_list)
        print("==============================================================================================")
        subprocess.run(subprocess_list)
