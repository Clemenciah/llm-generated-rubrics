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

from models.api_loader import load_api, qwen_api_scoring_json_mode
from data.data_loader import Dataset
from models.runners import end_to_end_scoring
from util.formatting_tools import ScoringParser, IdentityParser
from prompts.prompt_template import retrieve_set_of_tools
from models.metric import deserialize_metric
from results.save_and_load import save_scoring_results
from util.configuration import Configuration
import json
#################Main function######################
def scoring():
    parser = argparse.ArgumentParser(description='Run the specified runner with the given data and API.')
    parser.add_argument('--data', choices=["USR", "SummEval", "SumPubMed", "HelpSteer2"], default="USR", help='The name of the dataset to use')
    parser.add_argument('--api', choices=["gpt-4o","gpt-4o-mini","llama","vicuna","mixtral","qwen"], default="gpt-4o-mini", help='The LLM API to use')
    parser.add_argument('--use_system', action='store_true', help='Whether to use expert identity as the system prompt.')
    parser.add_argument('--use_sample', action='store_true', help='Whether to use samples with the pair response in the assessment.')
    parser.add_argument('--number_of_shots', type=int, default= 5, help='Number of sampels to use in the assessment prompt.')
    parser.add_argument('--use_temperature', action='store_true', help='If True, then a high temperature will be used and we will get major vote from their assessments.')
    parser.add_argument('--setting_name', type=str, required=True, help='The name of the set of metrics to load.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    print('Loading the configuration...')
    configuration_dict = {
        "model_name": args.api, 
        "use_system": args.use_system, 
        "use_sample": args.use_sample,
        "number_of_shots": args.number_of_shots,
        "use_temperature": args.use_temperature,
    }
    configuration = Configuration(**configuration_dict)
        
    print(f'Loading data: {args.data}')
    dataset = Dataset(datasets_dir, data_root_dir, args.data)
    
    print(f'Loading API: {args.api}')
    api_function = load_api(args.api)

    if  args.api == "qwen":
        api_function = qwen_api_scoring_json_mode

    print(f'Retrieving the essential set of prompting tooos for {args.api}')
    set_of_tools = retrieve_set_of_tools(args.api)
    
    print(f'loading the metrics list...')
    if args.setting_name == "ground_truth" or args.setting_name == "ground_truth_use_sample":
        metrics_list = [deserialize_metric(metric) for metric in json.load(open(f'data_information/{args.data}/ground_truth_metrics_set.json'))]
    else:
        metrics_list = [deserialize_metric(metric) for metric in json.load(open(f'results/instruction/{args.data}/instruction/{args.setting_name}'))]

    print('Loading the parser...')
    scoring_parser = ScoringParser(args.api).parser
    expert_identity_parser = IdentityParser(args.api).parser
    set_of_tools["scoring_parser"] = scoring_parser
    set_of_tools["expert_identity_parser"] = expert_identity_parser
    
    print('Assessment with the proposed set of metrics...')
    
    scored_samples = end_to_end_scoring(api_function,
                            args.api, 
                            set_of_tools,
                            dataset, 
                            metrics_list,
                            configuration,
                            seed = 42)
    save_scoring_results("scoring", scored_samples, args.data, args.api, f"by{args.api}.{args.setting_name}") # for when we want to use different LLM metrics

if __name__ == '__main__':
    scoring()