import subprocess
import os
import glob

# you can add other settings here
files = [
    'model_name=gpt-4o-mini-use_system=False-use_sample=False-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=gpt-4o-mini-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=gpt-4o-mini-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=gpt-4o-use_system=False-use_sample=False-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=gpt-4o-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=gpt-4o-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=llama-use_system=False-use_sample=False-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=llama-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=llama-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=mixtral-use_system=False-use_sample=False-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=mixtral-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=mixtral-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=qwen-use_system=False-use_sample=False-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=qwen-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json',
    'model_name=qwen-use_system=False-use_sample=True-use_response=True-number_of_shots=3-use_temperature=False-use_restriction=False.json'
]

dataset_names = ["USR","SummEval", "SumPubMed", "HelpSteer2"]

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
    for dataset in dataset_names:
        data = dataset
        subprocess_list = ["poetry", "run", "python", os.path.join(".", "scripts", "instructions.py"), "--data",
                           dataset, "--api", api, "--setting_name", json_file]
        print(subprocess_list)
        print("==============================================================================================")
        subprocess.run(subprocess_list)
