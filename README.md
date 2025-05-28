# Data Asset Annotation In Enterprise Knowledge Graphs

Data Asset Annotation consiste of annotating data object (in our case, tables and columns) with business concepts in an Enterprise Knowledge Graph (EKG) environment.

This application permits to:
1. Load Tabular Data ('t2dv2', 'zeenea-open-ds', 'turl-cta')
2. Create Syntactic (TF-IDF), Semantic (LM), and Graph Embeddings
3. Annotation Approaches:
    1. Random model
    2. Syntactic model
    3. Semantic model
    4. Binary classifier model
    5. Graph model
    6. Cross-similarity models (syntactic and/or semantic and/or graph similarities)
    7. Hybrid-embedding models (syntactic and/or semantic and/or graph embeddings)

The experimentations are tracked using MLFlow server.

## Quick Start
### Step 1: Create a Virtual Evironnement and install Requirements
```
python3 -m venv my-venv
source my-env/bin/activate
 
pip install -r requirements.txt
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
* __object_to_annotate__, of type str and takes value in ('column', 'dataset')
* __random_state_index__, of type int and takes value in (0, 1, 2)


#### Run specific models 
##### Unsupervised Models
1. To run Random model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_random_model
```
2. To run Syntactic model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --generate_syntactic_embeddings --enable_syntactic_model
```
3. To run Semantic model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --generate_semantic_embeddings --enable_semantic_model
```
#### Supervised Models
##### 1. To run Binary Classifier model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --generate_semantic_textual_links --enable_binary_classifier_model
```
##### 2. To run Graph model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_graph_model
```
##### 3. Cross-Similarity-based models 
###### 3.1. To run Cross Semantic and Syntactic Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_cross_model_syn_sem_similiarity_learning
```
###### 3.2. To run Cross Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_cross_model_sem_graph_similarity_learning
```
###### 3.3. To run Cross Syntactic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_cross_model_syn_graph_similarity_learning
```
###### 3.4. To run Cross Syntactic, Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_cross_model_syn_sem_graph_similarity_learning
```
##### 4. Hybrid-Embedding-based models
###### 3.1. To run Hybrid Semantic and Syntactic embeddings model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_hybrid_model_syn_sem_embedding_learning
```
###### 3.2. To run Hybrid Semantic and Graph Embeddings model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_hybrid_model_sem_graph_embedding_learning
```
###### 3.3. To run Hybrid Syntactic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_hybrid_model_syn_graph_embedding_learning
```
###### 3.4. To run Hybrid Syntactic, Semantic and Graph Similarity model:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index  --enable_hybrid_model_syn_sem_graph_embedding_learning
```
##### 5. Additiona ML Classifier models using Synatcic, Semantic and Graph similarities:
###### 5.1. To run Decision Tree model on Syntactic, Semantic and Graph similarities:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_decision_tree_classifier_model_syn_sem_graph_similarity_learning
```
###### 5.2. To run Random Forest Classifier on Syntactic, Semantic and Graph similarities:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_random_forest_classifier_model_syn_sem_graph_similarity_learning
```
###### 5.3. To run SVM Classifier on Syntactic, Semantic and Graph similarities:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_svm_classifier_model_syn_sem_graph_similarity_learning
```
###### 5.4. To run XGBoostClassifier on Syntactic, Semantic and Graph Similarities:
```
python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=0 --enable_xgboost_classifier_model_syn_sem_graph_similarity_learning
```


