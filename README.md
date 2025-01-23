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

`$bash launch.sh`



