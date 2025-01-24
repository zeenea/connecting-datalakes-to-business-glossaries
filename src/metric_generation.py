import pandas as pd
import json
import os

if __name__ == '__main__':

    metrics_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/metrics"

    model_dirs = [dir for dir in os.listdir(metrics_path) if not os.path.isfile(dir) ]

    metrics_df = pd.DataFrame(columns=['model', 'dataset_name', 'nb_epochs', 'random_state', 'mrr', 'hit@10'])

    for model_name in model_dirs:

        row_dict = {}
        row_dict['model'] = model_name

        for metric_file in os.listdir():

            with open(f"{metrics_path}/{model_name}/metric_file", 'r') as f:
                metric_tmp_dict_file = f.read()
                metric_tmp_dict = json.load(metric_tmp_dict_file)

        print(metric_tmp_dict)
        
    
    