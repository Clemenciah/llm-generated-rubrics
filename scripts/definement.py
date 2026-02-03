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
from models.runners import end_to_end_definement
from util.configuration import Configuration
from util.formatting_tools import DefinitionParser, IdentityParser
from prompts.prompt_template import retrieve_set_of_tools
#################Main function######################
def definement():
    parser = argparse.ArgumentParser(description='Run the specified runner with the given data and API.')
    parser.add_argument('--data', choices=["USR", "SummEval", "SumPubMed", "MT-Bench", "CommonGen", "HelpSteer2"], default="USR", help='The name of the dataset to use')
    parser.add_argument('--api', choices=["gpt-4o","gpt-4o-mini","llama","vicuna","mixtral","qwen"], default="gpt-4o-mini", help='The LLM API to use')
    parser.add_argument('--use_system', action='store_true', default=False, help='Whether to use system')
    parser.add_argument('--use_temperature', action='store_true', default=False, help='Whether to use temperature method in metric generation')
    parser.add_argument('--use_restriction', action='store_true', default=False, help='Whether to restrict the model to generate as much metrics as the default of the task')
    parser.add_argument('--use_sample', action='store_true', default=False, help='Whether to use samples in the definition prompt')
    parser.add_argument('--use_response', action='store_true', default=False, help='Whether to use responses in samples')
    parser.add_argument('--number_of_shots', type=int, default=3, help='Number of instances for difinition prompt')
    parser.add_argument('--use_similarity', action='store_true', default=False, help='Whether to use similarity')
    parser.add_argument('--similarity_threshold', type=float, default=0.9, help='Metrics similarity threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    print('Loading the configuration...')
    configuration_dict = {
        "model_name": args.api, 
        "use_system": args.use_system, 
        "use_sample": args.use_sample, 
        "use_response": args.use_response, 
        "number_of_shots": args.number_of_shots, 
        "use_temperature": args.use_temperature, 
        "use_restriction": args.use_restriction
    }
    configuration = Configuration(**configuration_dict)
    
    print(f'Loading data: {args.data}')
    dataset = Dataset(datasets_dir, data_root_dir, args.data)
    
    print(f'Loading API: {args.api}')
    api_function = load_api(args.api)
    
    print(f'Retrieving the essential set of prompting tooos for {args.api}')
    set_of_tools = retrieve_set_of_tools(args.api)
    
    print('Loading the parser...')
    definition_parser = DefinitionParser(args.api).parser
    identity_parser = IdentityParser(args.api).parser
    set_of_tools["expert_identity_parser"] = identity_parser
    set_of_tools["definement_parser"] = definition_parser
    
    
        
    print('Running the runner with the data and API...')
    
    metrics = end_to_end_definement(api_function, 
                                    set_of_tools,
                                    configuration,
                                    dataset, 
                                    seed = 42)
                                    
    
    print('Process completed successfully.')
    print('printing results of metrics definement...')
    for metrics_list in metrics:
        for metric in metrics_list:
            print(metric.to_str())

if __name__ == '__main__':
    definement()