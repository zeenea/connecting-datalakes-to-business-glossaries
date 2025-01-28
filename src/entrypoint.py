from link_prediction_utilities import load_data, semantic_model, graph_model, hybrid_model, hybrid_similarity_model
import argparse
import yaml
import logging
import os

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def starts_load_data():
        load_data_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/load_data_input_config.yaml"
        load_data_yaml = load_yaml(load_data_yaml_file_path)
        load_data_yaml_args = load_data_yaml.get("load_data_args", {})
        
        load_data_parser = argparse.ArgumentParser("Load Data Parser")
        load_data_parser.add_argument('--dataset_name', type=str)
        load_data_parser.add_argument('--object_to_predict', type=str)
        load_data_parser.add_argument('--random_state_index', type=int)
        load_data_parser.add_argument('--neg_strategy', type=str, default=load_data_yaml_args.get('neg_strategy'))
        
        load_data_args, _ = load_data_parser.parse_known_args()
        load_data.main(load_data_args)

def starts_semantic_model():
        semantic_model_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/semantic_model_input_config.yaml"
        semantic_model_yaml = load_yaml(semantic_model_yaml_file_path)
        semantic_model_yaml_args = semantic_model_yaml.get("semantic_model_args", {})
    
        sem_model_parser = argparse.ArgumentParser("Semantic Model Parser")
        sem_model_parser.add_argument('--dataset_name', type=str)
        sem_model_parser.add_argument('--object_to_predict', type=str)
        sem_model_parser.add_argument('--random_state_index', type=int)
        sem_model_parser.add_argument('--top_k', type=int, default=semantic_model_yaml_args.get('top_k'))
        sem_model_parser.add_argument('--nb_epochs', type=int, default=semantic_model_yaml_args.get('nb_epochs'))
        
        sem_model_args, _ = sem_model_parser.parse_known_args()
        semantic_model.main(sem_model_args)


def starts_graph_model():
        graph_model_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/graph_model_input_config.yaml"
        graph_model_yaml = load_yaml(graph_model_yaml_file_path)
        graph_model_yaml_args = graph_model_yaml.get("graph_model_args", {})
    
        graph_model_parser = argparse.ArgumentParser("Graph Model Parser")
        graph_model_parser.add_argument('--dataset_name', type=str)
        graph_model_parser.add_argument('--object_to_predict', type=str)
        graph_model_parser.add_argument('--random_state_index', type=int)
        graph_model_parser.add_argument('--max_epochs', type=int, default=graph_model_yaml_args.get('max_epochs'))
        graph_model_parser.add_argument('--graph_dim_embeddings', type=int, default=graph_model_yaml_args.get('graph_dim_embeddings'))
        graph_model_parser.add_argument('--learning_rate', type=float, default=graph_model_yaml_args.get('learning_rate'))
    
        graph_model_args, _ = graph_model_parser.parse_known_args()
        graph_model.main(graph_model_args)


def starts_hybrid_model():
    hybrid_model_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/hybrid_model_input_config.yaml"
    hybrid_model_yaml = load_yaml(hybrid_model_yaml_file_path)
    hybrid_model_yaml_args = hybrid_model_yaml.get("hybrid_model_args", {})

    hybrid_model_parser = argparse.ArgumentParser("Hybrid Model Parser")
    hybrid_model_parser.add_argument('--dataset_name', type=str)
    hybrid_model_parser.add_argument('--object_to_predict', type=str)
    hybrid_model_parser.add_argument('--random_state_index', type=int)
    hybrid_model_parser.add_argument('--batch_size', type=int, default=hybrid_model_yaml_args.get('batch_size'))
    hybrid_model_parser.add_argument('--num_workers', type=int, default=hybrid_model_yaml_args.get('num_workers'))
    hybrid_model_parser.add_argument('--nb_epochs', type=int, default=hybrid_model_yaml_args.get('nb_epochs'))
    hybrid_model_parser.add_argument('--num_classes', type=int, default=hybrid_model_yaml_args.get('num_classes'))
    hybrid_model_parser.add_argument('--learning_rate', type=float, default=hybrid_model_yaml_args.get('learning_rate'))
    hybrid_model_parser.add_argument('--hidden_layer_dim', type=int, default=hybrid_model_yaml_args.get('hidden_layer_dim'))
    hybrid_model_parser.add_argument('--top_k', type=int, default=hybrid_model_yaml_args.get('top_k'))

    hybrid_model_args, _ = hybrid_model_parser.parse_known_args()
    hybrid_model.main(hybrid_model_args)

def starts_hybrid_sim_model():
    hybrid_sim_model_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/hybrid_similarity_model_input_config.yaml"
    hybrid_sim_model_yaml = load_yaml(hybrid_sim_model_yaml_file_path)
    hybrid_sim_model_yaml_args = hybrid_sim_model_yaml.get("hybrid_similarity_model_args", {})

    hybrid_sim_model_parser = argparse.ArgumentParser("Hybrid Similarity Model Parser")
    hybrid_sim_model_parser.add_argument('--dataset_name', type=str)
    hybrid_sim_model_parser.add_argument('--object_to_predict', type=str)
    hybrid_sim_model_parser.add_argument('--random_state_index', type=int)
    hybrid_sim_model_parser.add_argument('--batch_size', type=int, default=hybrid_sim_model_yaml_args.get('batch_size'))
    hybrid_sim_model_parser.add_argument('--num_workers', type=int, default=hybrid_sim_model_yaml_args.get('num_workers'))
    hybrid_sim_model_parser.add_argument('--nb_epochs', type=int, default=hybrid_sim_model_yaml_args.get('nb_epochs'))
    hybrid_sim_model_parser.add_argument('--num_classes', type=int, default=hybrid_sim_model_yaml_args.get('num_classes'))
    hybrid_sim_model_parser.add_argument('--learning_rate', type=float, default=hybrid_sim_model_yaml_args.get('learning_rate'))
    hybrid_sim_model_parser.add_argument('--hidden_layer_dim', type=int, default=hybrid_sim_model_yaml_args.get('hidden_layer_dim'))
    hybrid_sim_model_parser.add_argument('--top_k', type=int, default=hybrid_sim_model_yaml_args.get('top_k'))


    hybrid_sim_model_parser_args, _ = hybrid_sim_model_parser.parse_known_args()
    hybrid_similarity_model.main(hybrid_sim_model_parser_args)


def are_embeddings_generated(dataset_name, model_name, random_state_index):
    
    random_state = [42, 84, 13][random_state_index]

    embedding_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings/dataset_name={dataset_name}/model_type={model_name}/random_state={random_state}"

    return os.path.exists(embedding_path)
        

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("--------------------- Link Prediction Model Selection ----------------------------")
    choice_module_parser = argparse.ArgumentParser("Choice Module Parser")
    choice_module_parser.add_argument('--enable_semantic_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_graph_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_hybrid_model', action='store_true', default=False)
    choice_module_parser.add_argument('--enable_hybrid_sim_model', action='store_true', default=False)
    choice_module_parser.add_argument('--dataset_name', type=str)
    choice_module_parser.add_argument('--random_state_index', type=int)
    
    choice_module_parser_args, unkown_args = choice_module_parser.parse_known_args()

    dataset_name = choice_module_parser_args.dataset_name
    random_state_index = choice_module_parser_args.random_state_index
    
    logger.info(choice_module_parser_args)
    logger.info(unkown_args)

    logger.info("--------------------- Starts load_data.py Module ---------------------------------")
    starts_load_data()

    if choice_module_parser_args.enable_semantic_model:
        logger.info("--------------------- Starts semantic_model.py Module ----------------------------")
        starts_semantic_model()

    if choice_module_parser_args.enable_graph_model:
        logger.info("--------------------- Starts graph_model.py Module -------------------------------")
        starts_graph_model()

    if choice_module_parser_args.enable_hybrid_model:
        if choice_module_parser_args.enable_graph_model or (are_embeddings_generated(dataset_name, 'graph-based', random_state_index ) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index)):
            logger.info("----------------------- Starts hybrid_model.py Module ----------------------------")
            starts_hybrid_model()
        else:
            logger.info("--------------------- Starts graph_model.py Module -------------------------------")
            starts_graph_model()
            
            logger.info("----------------------- Starts hybrid_model.py Module ----------------------------")
            starts_hybrid_model()
    
    if choice_module_parser_args.enable_hybrid_sim_model:
        if choice_module_parser_args.enable_graph_model or (are_embeddings_generated(dataset_name, 'graph-based', random_state_index ) and are_embeddings_generated(dataset_name, 'semantic-based', random_state_index)):
            logger.info("----------------------- Starts hybrid_sim_model.py Module ----------------------------")
            starts_hybrid_sim_model()
        else:
            logger.info("--------------------- Starts graph_model.py Module -------------------------------")
            starts_graph_model()
            
            logger.info("----------------------- Starts hybrid_sim_model.py Module ----------------------------")
            starts_hybrid_sim_model()


