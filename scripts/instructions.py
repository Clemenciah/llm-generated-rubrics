from dotenv import load_dotenv
import argparse
import os
import sys

###############Directories#########################
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

datasets_dir = os.path.join(parent_dir, 'datasets')
data_root_dir = os.path.join(parent_dir, 'data_information')

###############Loading api_key data################
load_dotenv()

from models.api_loader import load_api
from data.data_loader import Dataset
from models.runners import metric_instruction_generation
from util.formatting_tools import InstructionParser, IdentityParser
from prompts.prompt_template import retrieve_set_of_tools
from models.metric import deserialize_metric
from results.save_and_load import save_instruction_results
from util.similarity_tools import filter_names
import json
#################Main function######################
def instruction_generation():
    parser = argparse.ArgumentParser(description='Run the specified runner with the given data and API.')
    parser.add_argument('--data', choices=["USR", "SummEval", "SumPubMed", "MT-Bench", "CommonGen", "HelpSteer2"], default="USR", help='The name of the dataset to use')
    parser.add_argument('--api', choices=["gpt-4o","gpt-4o-mini","llama","vicuna","mixtral","qwen"], default="gpt-4o-mini", help='The LLM API to use')
    parser.add_argument('--setting_name', type=str, required=True, help='The name of the set of metrics to load.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
        
    print(f'Loading data: {args.data}')
    dataset = Dataset(datasets_dir, data_root_dir, args.data)
    
    print(f'Loading API: {args.api}')
    api_function = load_api(args.api)
    
    print('System prompt setting...')
    use_system = True if "use_system=True" in args.setting_name else False
    
    print(f'Retrieving the essential set of prompting tools for {args.api}')
    set_of_tools = retrieve_set_of_tools(args.api)
    
    print(f'loading the metrics list...')
    try:
        metrics_list = [deserialize_metric(metric) for metric in json.load(open(f'results/definition/{args.data}/{args.setting_name}'))]
    except:
        metrics_list = [deserialize_metric(metric) for metrics in json.load(open(f'results/definition/{args.data}/{args.setting_name}')) for metric in metrics] 
    metrics_list = filter_names(metrics_list)
    
    print('Loading the parser...')
    instruction_parser = InstructionParser(args.api).parser
    expert_identity_parser = IdentityParser(args.api).parser
    set_of_tools["instruction_parser"] = instruction_parser
    set_of_tools["expert_identity_parser"] = expert_identity_parser
    
    print('Running the runner with the data and API...')
    
    metrics = metric_instruction_generation(api_function,
                                            args.api, 
                                            set_of_tools,
                                            dataset, 
                                            metrics_list,
                                            use_system,
                                            seed = 42)
    save_instruction_results("instruction", metrics, args.data, args.setting_name)
    
    print('printing results of instruction generation...')
    for metric in metrics:
        print(metric.to_inst())
    if not os.path.exists(f"results/instruction/{args.data}/instruction/{args.api}_ground_truth.json"):
        ground_truth_metrics = dataset.ground_truth_metrics
        if args.data == "USR":
            ground_truth_metrics = ground_truth_metrics[:-1]
        metrics = metric_instruction_generation(api_function,
                                                args.api,
                                                set_of_tools,
                                                dataset,
                                                ground_truth_metrics,
                                                False,
                                                seed=42
                                                )
        if args.data == "USR":
            metrics.append(dataset.ground_truth_metrics[-1])
        save_instruction_results("instruction", metrics, args.data, f"{args.api}_ground_truth.json")
    
    print('Process completed successfully.')

if __name__ == '__main__':
    instruction_generation()