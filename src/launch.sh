#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=21600
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#module purge                        # Nettoyage de l'environnement
#module load ~/anaconda3             # Chargement du module anaconda3
#conda activate myenv                # Activation de votre environnement python
#conda init

dataset_name='zeenea-open-ds'
object_to_annotate='dataset'         # 'column' or 'dataset'
random_state_index=0                # 0 or 1 or 2

mlflow server --host 127.0.0.1 --port 8080 &

python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --generate_graph_embeddings --enable_graph_model
#--generate_syntactic_embeddings --enable_syntactic_model

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0  --enable_cross_model_syn_sem_graph_similarity_learning

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1  --enable_xgboost_classifier_model_syn_sem_graph_similarity_learning

