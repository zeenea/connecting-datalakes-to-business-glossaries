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
object_to_predict='column'
enable_semantic_model=False
enable_graph_model=False
enable_hybrid_model=False
enable_hybrid_sim_model=True

python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0 --enable_semantic_model=$enable_semantic_model --enable_graph_model=$enable_graph_model --enable_hybrid_model=$enable_hybrid_model enable_hybrid_sim_model=$enable_hybrid_sim_model   # Prediciton de lien
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1 --enable_semantic_model=$enable_semantic_model --enable_graph_model=$enable_graph_model --enable_hybrid_model=$enable_hybrid_model enable_hybrid_sim_model=$enable_hybrid_sim_model   # Prediciton de lien
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2 --enable_semantic_model=$enable_semantic_model --enable_graph_model=$enable_graph_model --enable_hybrid_model=$enable_hybrid_model enable_hybrid_sim_model=$enable_hybrid_sim_model  # Prediciton de lien
