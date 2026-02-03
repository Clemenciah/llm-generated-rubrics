from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

def merge_similar_metrics(metric_description_df, 
                          use_similarity, 
                          similarity_threshold,
                          meaningfulness_threshold = 2):
  if use_similarity:
    model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code = True)
    # Create embeddings for each of the metrics' descriptions
    metric_description_df['embeddings'] = metric_description_df['description'].apply(lambda x: model.encode(x))
    # From the embeddings and the defined threshold identifies the similar metrics
    similar_metrics_dict = find_similar_metrics(metric_description_df, threshold = similarity_threshold)
    # Merge the metrics with similar descriptions
    metric_description_df = replace_similar_metrics(metric_description_df, similar_metrics_dict)

    # Group metric_description_df by filename and count the occurrences of each name within each group
    name_counts = metric_description_df['name'].value_counts()


    name_counts_df = name_counts.to_frame(name='count').reset_index()

    # Filter the DataFrame to keep only names with more than three appearances for each group
    filtered_name_counts_df = name_counts_df[name_counts_df['count'] >= meaningfulness_threshold]


    names_to_keep = filtered_name_counts_df['name'].tolist()
    

    metric_description_df = metric_description_df.drop(columns=['embeddings'])
  else:
    metric_description_df = normalize_similar_names(metric_description_df, threshold = similarity_threshold)
    names_to_keep = metric_description_df['name'].unique()
    
  # Create a new dataframe with only the first row for each name in the list
  final_method = pd.DataFrame()
  for name in names_to_keep:
      row = metric_description_df[metric_description_df['name'] == name].iloc[0]
      final_method = pd.concat([final_method, pd.DataFrame([row])], ignore_index=True)

  return final_method


def find_similar_metrics(df, threshold=0.9):
    similar_metrics = {}
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
          if df["name"].iloc[i] != df["name"].iloc[j]:
            similarity = util.cos_sim(df['embeddings'].iloc[i], df['embeddings'].iloc[j]).item()
            if similarity >= threshold:
                if i not in similar_metrics:
                    similar_metrics[i] = []
                similar_metrics[i].append(j)
    return similar_metrics

def replace_similar_metrics(df, similar_metrics_dict):
  # Iterate over the dictionary in reverse order of keys
  for key, values in reversed(list(similar_metrics_dict.items())):
    dominant_metric = key

    for value in values:
        #replace all metric names and descriptions with the key metric
        df.loc[value, 'name'] = df.loc[key, 'name']
        df.loc[value, 'description'] = df.loc[key, 'description']
  return df

def filter_names(metrics_list):
  """
  Filters a metrics_list and removes redundancies based on metric names.
  """
  
  names_list = []
  list_of_metrics = []
  for metric in metrics_list:
    if metric.name not in names_list:
      names_list.append(metric.name)
      list_of_metrics.append(metric)
  
  return list_of_metrics

def normalize_similar_names(df, threshold=0.8):
    """
    Normalizes similar names in a dataframe by comparing string similarities
    and keeping only unique values considering different written formats.
    
    Args:
        df: pandas DataFrame with a 'name' column
        threshold: similarity threshold for considering names as similar
    
    Returns:
        DataFrame with normalized names
    """
    

    df_normalized = df.copy()
    

    unique_names = df_normalized['name'].unique()
    

    name_mappings = {}
    
    # Compare each name with every other name
    for i, name1 in enumerate(unique_names):
        if name1 not in name_mappings.values():
            for name2 in unique_names[i+1:]:
                if name2 not in name_mappings:
                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

                    if similarity >= threshold:
                        name_mappings[name2] = name1
    
    # Replace similar names with their mapped values
    df_normalized['name'] = df_normalized['name'].replace(name_mappings)
    
    return df_normalized
