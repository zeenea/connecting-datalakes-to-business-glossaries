from link_prediction_utilities import load_data, graph_model, hybrid_model
import argparse
import yaml
import logging


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("---------------------Starts load_data.py Module")
    
    load_data_yaml_file_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/src/input_yaml_config/load_data_input_config.yaml"
    load_data_yaml = load_yaml(load_data_yaml_file_path)
    load_data_yaml_args = load_data_yaml.get("load_data_args", {})
    
    load_data_parser = argparse.ArgumentParser("Load Data Parser")
    load_data_parser.add_argument('--dataset_name', type=str)
    load_data_parser.add_argument('--object_to_predict', type=str)
    load_data_parser.add_argument('--random_state_index', type=int)
    load_data_parser.add_argument('--neg_strategy', type=str, default=load_data_yaml_args.get('neg_strategy'))
    
    load_data_args = load_data_parser.parse_args()
    load_data.main(load_data_args)

    logger.info("---------------------Starts graph_model.py Module")
    
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

    graph_model_args = graph_model_parser.parse_args()
    graph_model.main(graph_model_args)

    
    logger.info("-----------------------Starts hybrid_model.py Module")
    
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

    hybrid_model_args = hybrid_model_parser.parse_args()
    hybrid_model.main(hybrid_model_args)
    



