import pandas as pd
import os
import pickle
from models.metric import deserialize_metric
import json

def load_data_info(info_dir, data_name):
    loading_dir = os.path.join(info_dir, data_name)
    ground_truth_metrics = [deserialize_metric(metric) for metric in json.load(open(os.path.join(loading_dir,'ground_truth_metrics_set.json'),'r'))]
    infos = {"dataset_name":data_name,
             "definement_task_description": pickle.load(open(os.path.join(loading_dir,"definement_task_description.pkl"),"rb")),
             "default_task_description": pickle.load(open(os.path.join(loading_dir,"default_task_description.pkl"),"rb")),
             "overall_score_column_name": pickle.load(open(os.path.join(loading_dir,"overall_score_column_name.pkl"),"rb")),
             "content_column_name": pickle.load(open(os.path.join(loading_dir,"content_column_name.pkl"),"rb")),
             "response_column_name": pickle.load(open(os.path.join(loading_dir,"response_column_name.pkl"), "rb")),
             "scoring_task_description": pickle.load(open(os.path.join(loading_dir,"scoring_task_description.pkl"),"rb")),
             "ground_truth_metrics": ground_truth_metrics
            }
    return infos

class Dataset:
    def __init__(self, data_dir, info_dir, data_name):
        self.load_data(data_dir, data_name)
        self.load_info(info_dir, data_name)

        
    def load_data(self, data_dir, data_name):
        self.data = pd.read_csv(os.path.join(data_dir,data_name)+".csv")
        
    def load_info(self, info_dir, data_name):
        information = load_data_info(info_dir, data_name)
        self.selected_subset = pd.read_csv(os.path.join(info_dir,data_name,"test.csv"))
        self.dataset_name = information["dataset_name"]
        self.definement_task_description = information["definement_task_description"]
        self.dataset_description = information["default_task_description"]
        self.overall_score_column_name = information["overall_score_column_name"]
        self.content_column_name = information["content_column_name"]
        self.response_column_name = information["response_column_name"]
        self.scoring_task_description = information["scoring_task_description"]
        self.ground_truth_metrics = information["ground_truth_metrics"]