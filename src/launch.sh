#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=21600
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

dataset_name='zeenea-open-ds'
object_to_annotate='dataset'         # 'column' or 'dataset'
#random_state_index=3

mlflow server --host 127.0.0.1 --port 8080 &

#source ../venv/bin/activate

#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --generate_semantic_embeddings --enable_semantic_model
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_graph_model 
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_hybrid_model_sem_graph_embedding_learning
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_cross_model_sem_graph_similarity_learning
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_reciprocal_rank_fusion_model

#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=1 --generate_semantic_embeddings --enable_semantic_model
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=1 --enable_graph_model 
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=1 --enable_hybrid_model_sem_graph_embedding_learning
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=1 --enable_cross_model_sem_graph_similarity_learning
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=1 --enable_reciprocal_rank_fusion_model

#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=2 --generate_semantic_embeddings --enable_semantic_model
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=2 --enable_graph_model 
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=2 --enable_hybrid_model_sem_graph_embedding_learning
#python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=2 --enable_cross_model_sem_graph_similarity_learning
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=2 --enable_reciprocal_rank_fusion_model

