import os
import json
from models.metric import serialize_metric

base_path = os.path.dirname(os.path.abspath(__file__))
def save_results(phase,
                 results, 
                 dataset_name,
                 setting_name
                 ):
        """
        Save the results to a JSON file.
        """
        save_path = os.path.join(base_path, phase, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        results = [serialize_metric(metric) for metrics in results for metric in metrics]
        json.dump(results, open(os.path.join(save_path, setting_name + ".json"),"w"),indent=4)
        # results.to_csv(f"{os.path.join(save_path, setting_name)}.csv")
        
def save_instruction_results(phase,
                            metrics, 
                            dataset_name,
                            setting_name
                            ):
        """
        Save the results to a JSON file.
        """
        save_path = os.path.join(base_path, phase, dataset_name,"instruction")
        os.makedirs(save_path, exist_ok=True)
        results = [serialize_metric(metric) for metric in metrics]
        json.dump(results, open(os.path.join(save_path, setting_name),"w"),indent=4)

def save_scoring_results(phase,
                         scored_samples,
                         dataset_name,
                         api_name,
                         metrics_source
                         ):
        """
        Save the results to a csv file.
        """
        save_path = os.path.join(base_path, phase, dataset_name, api_name)
        os.makedirs(save_path, exist_ok=True)
        scored_samples.to_csv(os.path.join(save_path, metrics_source + ".csv"), index=False)
        
def save_tagging_results(phase,
                         tags,
                         dataset_name,
                         saving_string
                         ):
        save_path = os.path.join(base_path, phase, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        json.dump(tags, open(os.path.join(save_path, saving_string),"w"),indent=4)