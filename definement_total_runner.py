import subprocess
import os

dataset_names = ["SummEval", "SumPubMed", "HelpSteer2"]

settings = [ # use or introduce the settings that you like
    # {
    #     'model_name': 'gpt-4o-mini',
    #     'use_system': False,
    #     'use_sample': False,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': True
    # },
    # {
    #     'model_name': 'gpt-4o',
    #     'use_system': False,
    #     'use_sample': False,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': True
    # },
    # {
    #     'model_name': 'llama',
    #     'use_system': False,
    #     'use_sample': True,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # },
    # {
    #     'model_name': 'mixtral',
    #     'use_system': False,
    #     'use_sample': True,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': True
    # },
    # {
    #     'model_name': 'qwen',
    #     'use_system': False,
    #     'use_sample': True,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # },
    # {
    #     'model_name': 'gpt-4o-mini',
    #     'use_system': True,
    #     'use_sample': True,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # },
    # {
    #     'model_name': 'gpt-4o',
    #     'use_system': True,
    #     'use_sample': True,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # },
    # {
    #     'model_name': 'llama',
    #     'use_system': False,
    #     'use_sample': False,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': True
    # },
    # {
    #     'model_name': 'mixtral',
    #     'use_system': False,
    #     'use_sample': False,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # },
    # {
    #     'model_name': 'qwen',
    #     'use_system': False,
    #     'use_sample': False,
    #     'use_response': False,
    #     'number_of_shots': 3,
    #     'use_temperature': False,
    #     'use_restriction': False
    # }

    {
        'model_name': 'gpt-4o-mini',
        'use_system': False,
        'use_sample': False,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'gpt-4o-mini',
        'use_system': False,
        'use_sample': True,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'gpt-4o-mini',
        'use_system': False,
        'use_sample': True,
        'use_response': True,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'gpt-4o',
        'use_system': False,
        'use_sample': False,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'gpt-4o',
        'use_system': False,
        'use_sample': True,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'gpt-4o',
        'use_system': False,
        'use_sample': True,
        'use_response': True,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'llama',
        'use_system': False,
        'use_sample': False,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'llama',
        'use_system': False,
        'use_sample': True,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'llama',
        'use_system': False,
        'use_sample': True,
        'use_response': True,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'mixtral',
        'use_system': False,
        'use_sample': False,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'mixtral',
        'use_system': False,
        'use_sample': True,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'mixtral',
        'use_system': False,
        'use_sample': True,
        'use_response': True,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'qwen',
        'use_system': False,
        'use_sample': False,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'qwen',
        'use_system': False,
        'use_sample': True,
        'use_response': False,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    },
    {
        'model_name': 'qwen',
        'use_system': False,
        'use_sample': True,
        'use_response': True,
        'number_of_shots': 3,
        'use_temperature': False,
        'use_restriction': False
    }
]

for dataset in dataset_names:
    for setting in settings:

        subprocess_list = [
            "poetry", "run", "python", os.path.join(".", "scripts", "definement.py"),
            "--data", dataset,
            "--api", setting["model_name"],
            "--number_of_shots", str(setting["number_of_shots"]),
        ]


        if setting["use_system"]:
            subprocess_list.append("--use_system")
        if setting["use_temperature"]:
            subprocess_list.append("--use_temperature")
        if setting["use_restriction"]:
            subprocess_list.append("--use_restriction")


        if setting["use_sample"]:
            subprocess_list.append("--use_sample")
            if setting["use_response"]:
                subprocess_list.append("--use_response")


        print("Executing:", " ".join(subprocess_list))
        print("=" * 90)
        subprocess.run(subprocess_list)
