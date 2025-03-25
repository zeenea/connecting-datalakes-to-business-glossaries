import pandas as pd
import json
import os
import ast

if __name__ == '__main__':

    metrics_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/metrics"

    model_dirs = [dir for dir in os.listdir(metrics_path) if not os.path.isfile(dir) and '.' not in dir ]

    #metrics_df = pd.DataFrame(columns=['model', 'dataset_name', 'nb_epochs', 'random_state', 'mrr', 'hit@10'])
    print(f"models_dir: {model_dirs}")

    all_metrics_df = pd.DataFrame()
    
    for model_name in model_dirs:
        print(f"model name: {model_name}")
        
        for metric_file in os.listdir(f"{metrics_path}/{model_name}"):
            
            if os.path.isfile(f"{metrics_path}/{model_name}/{metric_file}"):
                print(f"metric_file: {metric_file}")
                
                with open(f"{metrics_path}/{model_name}/{metric_file}", 'r') as f:
                    dict_str = f.read()
                    metric_dict_df = ast.literal_eval(dict_str)
                    metric_dict_df['model'] = model_name

                    if "column" in metric_file:
                        metric_dict_df['object_to_predict'] = 'column'
                    else:
                        metric_dict_df['object_to_predict'] = 'dataset'

                    all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame(metric_dict_df, index=[0])], axis=0)
                    all_metrics_df = all_metrics_df.reset_index(drop=True)
                    
        
    print(all_metrics_df)
    print
    all_metrics_df.to_csv(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_metrics_with_random_stats.csv")
    all_metrics_df['MRR'] = all_metrics_df['MRR'].astype(float)
    all_metrics_df['Hit@10'] = all_metrics_df['Hit@10'].astype(float)
    metrics_df = all_metrics_df.groupby(['dataset_name', 'object_to_predict', 'model', 'epochs']).agg({'MRR':['mean', 'std'], 'Hit@10':['mean', 'std']}).reset_index()
    metrics_df.columns = ['dataset_name', 'object_to_predict', 'model', 'epochs', 'MRR-mean', 'MRR-std', 'Hit@10-mean', 'Hit@10-std']
    
    metrics_df.to_csv(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_metrics.csv")
    col_metrics_df = metrics_df[metrics_df['object_to_predict']=='column'].sort_values(by=['dataset_name', 'object_to_predict', 'Hit@10-mean'], ascending=False)
    ds_metrics_df = metrics_df[metrics_df['object_to_predict']=='dataset'].sort_values(by=['dataset_name', 'object_to_predict', 'Hit@10-mean'], ascending=False)
    print(col_metrics_df)
    print(ds_metrics_df)
    col_metrics_df.to_csv(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_column_metrics.csv")
    ds_metrics_df.to_csv(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_dataset_metrics.csv")

    col_metrics_df['MRR'] = col_metrics_df.apply(lambda x: f"{round(x['MRR-mean'], 2)} ± {round(x['MRR-std'], 2)}", axis=1) 
    col_metrics_df['Hit@10'] = col_metrics_df.apply(lambda x: f"{round(x['Hit@10-mean'], 2)} ± {round(x['Hit@10-std'], 2)}", axis=1)

    ds_metrics_df['MRR'] = ds_metrics_df.apply(lambda x: f"{round(x['MRR-mean'], 2)} ± {round(x['MRR-std'], 2)}", axis=1) 
    ds_metrics_df['Hit@10'] = ds_metrics_df.apply(lambda x: f"{round(x['Hit@10-mean'], 2)} ± {round(x['Hit@10-std'], 2)}", axis=1)

    col_metrics_df = col_metrics_df[['dataset_name', 'model', 'MRR', 'Hit@10']]
    ds_metrics_df = ds_metrics_df[['dataset_name', 'model', 'MRR', 'Hit@10']]

    #col_metrics_df = col_metrics_df.set_index('model')
    #ds_metrics_df = ds_metrics_df.set_index('model')

    print(col_metrics_df)
    print(ds_metrics_df)
    
    with open(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_column_metrics.tex", 'w') as f:
        f.write(col_metrics_df.to_latex(index=False))

    with open(f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/final_results/model_dataset_metrics.tex", 'w') as f:
        f.write(ds_metrics_df.to_latex(index=False))

    

    

    
        
    
    