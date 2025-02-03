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

dataset_name='zeenea-open-ds'
object_to_predict='dataset'


python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0 --generate_semantic_textual_link_embeddings 

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --generate_syntactic_embeddings --enable_hybrid_model_syn_graph_embedding_learning

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --generate_syntactic_embeddings --enable_hybrid_model_syn_graph_embedding_learning

#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --generate_semantic_embeddings --generate_syntactic_embeddings --enable_cross_model_syn_sem_graph_embedding_learning
#python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --generate_semantic_embeddings --generate_syntactic_embeddings --enable_cross_model_syn_sem_graph_embedding_learning
