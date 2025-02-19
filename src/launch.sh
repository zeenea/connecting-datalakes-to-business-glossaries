#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=21600
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge                        # Nettoyage de l'environnement
#module load ~/anaconda3             # Chargement du module anaconda3
eval "$(conda shell.bash hook)"     # Initialisation du shell pour conda
conda activate myenv                # Activation de votre environnement python

dataset_name='turl-cta'
object_to_predict='dataset'


mlflow server --host 127.0.0.1 --port 8080 &

python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0  --enable_graph_model

python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1  --enable_graph_model

python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2  --enable_graph_model
#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0 --generate_semantic_textual_links --enable_binary_classifier_model 

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --generate_semantic_textual_links --enable_binary_classifier_model 

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --generate_semantic_textual_links --enable_binary_classifier_model 

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --generate_syntactic_embeddings --enable_hybrid_model_syn_graph_embedding_learning

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --generate_syntactic_embeddings --enable_hybrid_model_syn_graph_embedding_learning

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --generate_semantic_embeddings --generate_syntactic_embeddings --enable_cross_model_syn_sem_graph_embedding_learning
#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --generate_semantic_embeddings --generate_syntactic_embeddings --enable_cross_model_syn_sem_graph_embedding_learning
