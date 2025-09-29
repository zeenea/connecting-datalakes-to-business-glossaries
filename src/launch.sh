#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=21600
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source venv/bin/activate

dataset_name='turl-cta'
object_to_annotate='column'         # 'column' or 'dataset'
random_state_index=0                # 0 or 1 or 2

mlflow server --host 127.0.0.1 --port 8080 &

python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0  --enable_graph_model --generate_semantic_embeddings --enable_hybrid_model_sem_graph_embedding_learning

