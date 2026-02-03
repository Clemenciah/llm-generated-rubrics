import subprocess
import os
import glob

# Search for all .json files within results/definition/USR and extract only the filenames
#files = [os.path.basename(file) for file in glob.glob(os.path.join("results", "definition", "SumPubMed", "*.json"))]
files = ["model_name=mixtral-use_system=False-use_sample=True-use_response=False-number_of_shots=3-use_temperature=False-use_restriction=False.json"]


dataset_names = ["HelpSteer2"]

results_dir = os.path.join("results", "tag")

for json_file in files:
        for dataset in dataset_names:
            data = dataset
            subprocess_list = ["poetry", "run", "python", os.path.join(".", "scripts", "tagging.py"), "--data", dataset, "--setting_name", json_file]
            print(subprocess_list)
            print("==============================================================================================")
            subprocess.run(subprocess_list)
