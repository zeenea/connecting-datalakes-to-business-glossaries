# Link Prediction for Datasets and Fields with Business Glossary Items

The application is composed of 3 modules.

1. Data Loading and Semantic Embeddings Generation
2. Hetero-Graph Link Prediction and Graph Embeddings Generation
3. Hybrid Link Prediction using Semantic and Graph Embeddings


## Requirements should be installed before executing the application.

`pip install -r link-prediction-local/requirements.txt`

## Parameters and Hyperparameters for training the algorithms are defined in Yaml files as input config files.

`link-prediction-local/src/input_yaml_config/load_data_input_config.yaml`

`link-prediction-local/src/input_yaml_config/graph_model_input_config.yaml`

`link-prediction-local/src/input_yaml_config/hybrid_model_input_config.yaml`

## To run the application.

`$bash link-prediction-local/src/launch.sh`

## where we execute the entrypoint.py file with some required arguments as:
* dataset_name, of type str and takes value in ('t2dv2', 'zeenea-open-ds', 'turl-cta')
* object_to_predict, of type str and takes value in ('column', 'dataset')
* random_state_index, of type int and takes value in (0, 1, 2)

`python entrypoint --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index`

## To generate semantic embeddings or syntactic embeddings use 

`--generate_syntactic_embeddings` or `--generate_semantic_embeddings`

## To enable some models 



