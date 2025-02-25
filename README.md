# Link Prediction for Datasets and Fields with Business Glossary Items

The application is composed of 3 modules.

1. Data Loading and Semantic Embeddings Generation
2. Hetero-Graph Link Prediction and Graph Embeddings Generation
3. Hybrid Link Prediction using Semantic and Graph Embeddings


## Quick Start
### Step 1: Requirements
```
pip install -r link-prediction-local/requirements.txt
```


### step 2: Run
#### Parameters and Hyperparameters
Parameters and hyperparameters are defined for each model in the `model_configs.yaml` file.
  
`link-prediction-local/src/input_yaml_config/model_configs.yaml`


#### Application run
```
mlflow server --host 127.0.0.1 --port 8080
bash launch.sh
```

In `launch.sh` we execute the `entrypoint.py` file with some required arguments as:
* __dataset_name__, of type str and takes value in ('t2dv2', 'zeenea-open-ds', 'turl-cta')
* __object_to_predict__, of type str and takes value in ('column', 'dataset')
* __random_state_index__, of type int and takes value in (0, 1, 2)


#### Run specific models 
##### Unsupervised Models
1. To run Random model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_random_model
```
2. To run Syntactic model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --generate_syntactic_embeddings --enable_syntactic_model
```
3. To run Semantic model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index --generate_semantic_embeddings --enable_semantic_model
```
#### Supervised Models
##### 1. To run Binary Classifier model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index --generate_semantic_textual_links --enable_binary_classifier_model
```
##### 2. To run Graph model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index --enable_graph_model
```
##### 3. Cross-Similarity-based models 
###### 3.1. To run Cross Semantic and Syntactic Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_cross_model_syn_sem_similiarity_learning
```
###### 3.2. To run Cross Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_cross_model_sem_graph_similarity_learning
```
###### 3.3. To run Cross Syntactic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_cross_model_syn_graph_similarity_learning
```
###### 3.4. To run Cross Syntactic, Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_cross_model_syn_sem_graph_similarity_learning
```
##### 4. Hybrid-Embedding-based models
###### 3.1. To run Hybrid Semantic and Syntactic embeddings model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_hybrid_model_syn_sem_embedding_learning
```
###### 3.2. To run Hybrid Semantic and Graph Embeddings model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_hybrid_model_sem_graph_embedding_learning
```
###### 3.3. To run Hybrid Syntactic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_hybrid_model_syn_graph_embedding_learning
```
###### 3.4. To run Hybrid Syntactic, Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_predict=$object_to_predict --random_state_index=$random_state_index  --enable_hybrid_model_syn_sem_graph_embedding_learning
```


