# Link Prediction for Datasets and Fields with Business Glossary Items

The application is composed of 3 modules.

1. Data Loading and Semantic Embeddings Generation
2. Hetero-Graph Link Prediction and Graph Embeddings Generation
3. Hybrid Link Prediction using Semantic and Graph Embeddings


## Requirements should be installed before executing the application.

`pip install -r link-prediction-local/requirements.txt`

## Parameters and Hyperparameters for training the algorithms are defined in Yaml files as input config files.

`link-prediction-local/src/input_yaml_config/model_configs.yaml`


## To run the application.
`$ mlflow server --host 127.0.0.1 --port 8080`

`$ bash launch.sh`

## where we execute the entrypoint.py file with some required arguments as:
* __dataset_name__, of type str and takes value in ('t2dv2', 'zeenea-open-ds', 'turl-cta')
* __object_to_predict__, of type str and takes value in ('column', 'dataset')
* __random_state_index__, of type int and takes value in (0, 1, 2)

`python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index`

## To generate semantic embeddings, syntactic embeddings or graph embeddings use :

`--generate_syntactic_embeddings` or `--generate_semantic_embeddings` or `--generate_semantic_textual_links`

## To enable some models 
1. Starts random model: --enable_random_model
2. Starts syntactic model: --generate_syntactic_embeddings --enable_syntactic_model
3. Starts semantic model: --generate_semantic_embeddings --enable_semantic_model
4. Starts binary classifier model: --generate_semantic_textual_links --enable_binary_classifier_model
5. Starts graph model: --enable_graph_model
6. Starts cross semantic and syntactic similarity learning model: --enable_cross_model_syn_sem_similiarity_learning
7. Starts cross semantic and graph similarity learning model: --enable_cross_model_sem_graph_similarity_learning
8. Starts cross syntactic and graph similarity learning model: --enable_cross_model_syn_graph_similarity_learning
9. 
10. 



