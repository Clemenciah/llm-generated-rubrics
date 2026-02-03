from dotenv import load_dotenv
import argparse
import os
import sys
import json

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
from util.formatting_tools import gpt_tag_parser
from prompts.prompt_template import AllMetricsTags
from models.metric import deserialize_metric
from util.similarity_tools import filter_names
from models.runners import tagger_runner


#################Main function######################
def tag():
    parser = argparse.ArgumentParser(description='Run the specified runner with the given data and API.')
    parser.add_argument('--data', choices=["USR", "SummEval", "SumPubMed", "MT-Bench", "CommonGen", "HelpSteer2"],
                        default="USR", help='The name of the dataset to use')
    parser.add_argument('--setting_name', help='The name of the setting to compare the results of to the ground_truth.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(f'Loading ground truth metrics: {args.data}')
    dataset = Dataset(datasets_dir, data_root_dir, args.data)
    ground_truth_metrics = dataset.ground_truth_metrics
    if args.data == "USR":
        ground_truth_metrics = ground_truth_metrics[:-1]
    if args.data == "SummEval":
        ground_truth_metrics = ground_truth_metrics[:-1]
    if args.data == "SumPubMed":
        ground_truth_metrics = ground_truth_metrics[:-1]

    print(f'Loading API: gpt-4o')
    api_function = load_api("gpt-4o")

    print(f'loading the metrics list...')
    try:
        metrics_list = [deserialize_metric(metric) for metric in
                        json.load(open(f'results/definition/{args.data}/{args.setting_name}'))]
    except:
        metrics_list = [deserialize_metric(metric) for metrics in
                        json.load(open(f'results/definition/{args.data}/{args.setting_name}')) for metric in metrics]
    metrics_list = filter_names(metrics_list)

    print(f'Retrieving the essential set of prompting tools...')
    tag_formatter = AllMetricsTags

    print('Loading the parser...')
    tag_parser = gpt_tag_parser

    print('Running the runner with the data and API...')
    tags = tagger_runner(api_function,
                         metrics_list,
                         args.data,
                         "Generate their ideal responses/continuation to a conversation, considering the context and an interesting fact provided.",
                         ground_truth_metrics,
                         tag_formatter,
                         tag_parser,
                         args.setting_name,
                         seed=42)

    print('Process completed successfully.')
    print('printing results of metrics definement...')
    print('ground truth metrics names:')
    for metric in ground_truth_metrics:
        print(metric.name)

    print("=" * 100 + "\nUnder_Examine_metrics_and_tags:")
    for metric_tag in tags:
        print(metric_tag['name'], " : ", metric_tag['tags'])


if __name__ == '__main__':
    tag()
