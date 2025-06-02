#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=21600
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

dataset_name='zeenea-open-ds'
object_to_annotate='dataset'         # 'column' or 'dataset'
random_state_index=0                # 0 or 1 or 2

mlflow server --host 127.0.0.1 --port 8080 &

python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_svm_classifier_model_syn_sem_graph_similarity_learning

