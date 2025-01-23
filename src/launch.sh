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
object_to_predict='column'
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=0    # Prediciton de lien
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=1    # Prediciton de lien
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=2    # Prediciton de lien
