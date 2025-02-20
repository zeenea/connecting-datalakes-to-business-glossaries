from utilities import load_data
from models import semantic_model, syntactic_model, random_model, graph_model, hybrid_model_sem_graph_embedding_learning, hybrid_model_syn_sem_embedding_learning, cross_model_syn_sem_graph_embedding_learning, cross_model_sem_graph_similarity_learning, cross_model_syn_graph_similarity_learning, cross_model_syn_sem_similarity_learning, hybrid_model_syn_graph_embedding_learning, cross_model_syn_sem_graph_similarity_learning, binary_classifier_model

import argparse
import yaml
import logging
import os
import mlflow

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def starts_load_data():
        yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
        yaml_file = load_yaml(yaml_file_path)
        yaml_args = yaml_file.get("load_data_args", {})
        
        parser = argparse.ArgumentParser("Load Data Parser")
        parser.add_argument('--dataset_name', type=str)
        parser.add_argument('--object_to_predict', type=str)
        parser.add_argument('--random_state_index', type=int)
        parser.add_argument('--neg_strategy', type=str, default=yaml_args.get('neg_strategy'))
        parser.add_argument('--generate_syntactic_embeddings', action='store_true', default=False)
        parser.add_argument('--generate_semantic_embeddings', action='store_true', default=False)
        parser.add_argument('--generate_semantic_textual_links', action='store_true', default=False)
        
        args, _ = parser.parse_known_args()
        load_data.main(args)

def starts_random_model():    
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("random_model_args", {})

    parser = argparse.ArgumentParser("Random Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    random_model.main(args)

def starts_syntactic_model():

    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("syntactic_model_args", {})

    parser = argparse.ArgumentParser("Syntactic Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    
    args, _ = parser.parse_known_args()
    syntactic_model.main(args)


def starts_semantic_model():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("semantic_model_args", {})

    parser = argparse.ArgumentParser("Semantic Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    
    args, _ = parser.parse_known_args()
    semantic_model.main(args)


def starts_graph_model():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("graph_model_args", {})

    parser = argparse.ArgumentParser("Graph Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--max_epochs', type=int, default=yaml_args.get('max_epochs'))
    parser.add_argument('--graph_dim_embeddings', type=int, default=yaml_args.get('graph_dim_embeddings'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))

    args, _ = parser.parse_known_args()
    graph_model.main(args)


def starts_hybrid_model_sem_graph_embedding_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("hybrid_model_sem_graph_embed_learn_args", {})

    parser = argparse.ArgumentParser("Hybrid Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    hybrid_model_sem_graph_embedding_learning.main(args)

def starts_hybrid_model_syn_sem_embedding_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml = load_yaml(yaml_file_path)
    yaml_args = yaml.get("hybrid_model_syn_sem_embed_learn_args", {})

    parser = argparse.ArgumentParser("Hybrid Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    hybrid_model_syn_sem_embedding_learning.main(args)

def starts_hybrid_model_syn_graph_embedding_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml = load_yaml(yaml_file_path)
    yaml_args = yaml.get("hybrid_model_syn_graph_embed_learn_args", {})

    parser = argparse.ArgumentParser("Hybrid Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    hybrid_model_syn_graph_embedding_learning.main(args)
    
def starts_cross_model_syn_sem_graph_embedding_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("cross_model_syn_sem_graph_embed_learn_args", {})

    parser = argparse.ArgumentParser("Hybrid Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    cross_model_syn_sem_graph_embedding_learning.main(args)

def starts_cross_model_sem_graph_similarity_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("cross_model_sem_graph_similarity_learning_args", {})

    parser = argparse.ArgumentParser("Cross Sem Graph Similarity Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))


    parser_args, _ = parser.parse_known_args()
    cross_model_sem_graph_similarity_learning.main(parser_args)

def starts_cross_model_syn_sem_similarity_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("cross_model_syn_sem_similarity_learning_args", {})

    parser = argparse.ArgumentParser("Cross Syntactic Semantic Similarity Learning  Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))


    parser_args, _ = parser.parse_known_args()
    cross_model_syn_sem_similarity_learning.main(parser_args)

def starts_cross_model_syn_graph_similarity_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("cross_model_syn_graph_similarity_learning_args", {})

    parser = argparse.ArgumentParser("Cross Syntactic Graph Similarity Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))

    args, _ = parser.parse_known_args()
    cross_model_syn_graph_similarity_learning.main(args)

def starts_cross_model_syn_sem_graph_similarity_learning():
    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("cross_model_syn_sem_graph_similarity_learning_args", {})

    parser = argparse.ArgumentParser("Hybrid Syntactic Semantic Similarity Learning  Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('nb_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--hidden_layer_dim', type=int, default=yaml_args.get('hidden_layer_dim'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))


    parser_args, _ = parser.parse_known_args()
    cross_model_syn_sem_graph_similarity_learning.main(parser_args)


def starts_binary_classifier_model():

    yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("binary_classifier_model_args", {})

    parser = argparse.ArgumentParser("Binary Classifier Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_predict', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--max_epochs', type=int, default=yaml_args.get('max_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--learning_rate', type=float, default=yaml_args.get('learning_rate'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))
    
    args, _ = parser.parse_known_args()
    binary_classifier_model.main(args)
    
def are_embeddings_generated(dataset_name, model_name, random_state_index):
    
    random_state = [42, 84, 13][random_state_index]

    embedding_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings/dataset_name={dataset_name}/model_type={model_name}/random_state={random_state}"

    return os.path.exists(embedding_path)
        

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("--------------------- Link Prediction Model Selection ----------------------------")
    choice_module_parser = argparse.ArgumentParser("Choice Module Parser")
    
    choice_module_parser.add_argument('--enable_random_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_syntactic_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_semantic_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_graph_model', action='store_true', default=False)
    
    choice_module_parser.add_argument('--enable_hybrid_model_sem_graph_embedding_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_hybrid_model_syn_sem_embedding_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_hybrid_model_syn_graph_embedding_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_cross_model_syn_sem_graph_embedding_learning', action='store_true', default=False)
    
    choice_module_parser.add_argument('--enable_cross_model_sem_graph_similarity_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_cross_model_syn_sem_similarity_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_cross_model_syn_graph_similarity_learning', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_cross_model_syn_sem_graph_similarity_learning', action='store_true', default=False)

    choice_module_parser.add_argument('--enable_binary_classifier_model', action='store_true', default=False)
    
    choice_module_parser.add_argument('--dataset_name', type=str)
    choice_module_parser.add_argument('--random_state_index', type=int)
    
    choice_module_parser_args, unkown_args = choice_module_parser.parse_known_args()

    dataset_name = choice_module_parser_args.dataset_name
    random_state_index = choice_module_parser_args.random_state_index
    
    logger.info(choice_module_parser_args)
    logger.info(unkown_args)

    logger.info("--------------------- Starts load_data.py Module -------------------------------------------")
    starts_load_data()

    if choice_module_parser_args.enable_random_model:
        logger.info("--------------------- Starts random_model.py Module ------------------------------------")
        starts_random_model()

    if choice_module_parser_args.enable_syntactic_model:
        logger.info("--------------------- Starts syntactic_model.py Module ---------------------------------")
        starts_syntactic_model()
        
    if choice_module_parser_args.enable_semantic_model:
        logger.info("--------------------- Starts semantic_model.py Module ----------------------------------")
        starts_semantic_model()

    if choice_module_parser_args.enable_graph_model:
        logger.info("--------------------- Starts graph_model.py Module -------------------------------------")
        starts_graph_model()

    
    
    if choice_module_parser_args.enable_hybrid_model_syn_sem_embedding_learning:
        if are_embeddings_generated(dataset_name, 'semantic-based', random_state_index) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index):
            logger.info("---------------------- Starts hybrid_model_syn_sem_embedding_learning.py Module -------------")
            starts_hybrid_model_syn_sem_embedding_learning()
        else:
            logger.info("Error: Semantic and Syntactic Embeddings should be generated.")

    if choice_module_parser_args.enable_hybrid_model_sem_graph_embedding_learning:
        if (are_embeddings_generated(dataset_name, 'graph-based', random_state_index ) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index)):
            logger.info("----------------------- Starts hybrid_model_sem_graph_embedding_learning.py Module ------------------------------")
            starts_hybrid_model_sem_graph_embedding_learning()
        else:
            logger.info("--------------------- Starts graph_model.py Module ---------------------------------")
            starts_graph_model()
            
            logger.info("----------------------- Starts hybrid_model_sem_graph_embedding_learning.py Module ------------------------------")
            starts_hybrid_model_sem_grah_embedding_learning()
            
    if choice_module_parser_args.enable_hybrid_model_syn_graph_embedding_learning:
        if are_embeddings_generated(dataset_name, 'syntactic-based', random_state_index) and are_embeddings_generated(dataset_name, 'graph-based', random_state_index):
            logger.info('------------------------- Starts hybrid_model_syn_graph_embedding_learning.py Module ----')
            starts_hybrid_model_syn_graph_embedding_learning()
        else:
            logger.info("Error: Syntactic and Graph Embeddings should be generated.")

        
    if choice_module_parser_args.enable_cross_model_syn_sem_graph_embedding_learning:
        if are_embeddings_generated(dataset_name, 'semantic-based', random_state_index) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index) and are_embeddings_generated(dataset_name, 'graph-based', random_state_index):
            logging.info("---------------------- Starts cross_model_syn_sem_graph_embedding_learning.py Module ------")
            starts_cross_model_syn_sem_graph_embedding_learning()
        else:
            logger.info("Semantic, Syntactic and Graph Embeddings should be generated.")
    

    
    if choice_module_parser_args.enable_cross_model_sem_graph_similarity_learning:
        if are_embeddings_generated(dataset_name, 'graph-based', random_state_index ) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index):
            logger.info("----------------------- Starts cross_model_sem_graph_similarity_learning.py Module --------------------------")
            starts_cross_model_sem_graph_similarity_learning()
        else:
            logger.info("--------------------- Starts graph_model.py Module ---------------------------------")
            starts_graph_model()
            
            logger.info("----------------------- Starts cross_model_sem_graph_similarity_learning.py Module --------------------------")
            starts_cross_model_sem_graph_similarity_learning()

    if choice_module_parser_args.enable_cross_model_syn_sem_similarity_learning:
        if are_embeddings_generated(dataset_name, 'syntactic-based', random_state_index) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index):
            logger.info("------------------------ Starts cross_model_syn_sem_similarity_learning.py Module --")
            starts_cross_model_syn_sem_similarity_learning()
        else:
            logger.info("Error: Syntactic and Semantic Embeddings should be generated.")

    if choice_module_parser_args.enable_cross_model_syn_graph_similarity_learning:
        if are_embeddings_generated(dataset_name, 'syntactic-based', random_state_index) and are_embeddings_generated(dataset_name, 'graph-based', random_state_index):
            logger.info("------------------------ Starts cross_model_syn_graph_similarity_learning.py Module --")
            starts_cross_model_syn_graph_similarity_learning()
        else:
            logger.info("Error: Syntactic and Graph Embeddings should be generated.")

    if choice_module_parser_args.enable_cross_model_syn_sem_graph_similarity_learning:
        if  are_embeddings_generated(dataset_name, 'semantic-based', random_state_index) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index) and are_embeddings_generated(dataset_name, 'graph-based', random_state_index):
            logger.info("------------------------ Starts cross_model_syn_sem_graph_similarity_learning.py Module --")
            starts_cross_model_syn_sem_graph_similarity_learning()


    if choice_module_parser_args.enable_binary_classifier_model:
        logger.info("---------------------------- Starts binary_classifier_model.py Module -----------------------")
        starts_binary_classifier_model()


