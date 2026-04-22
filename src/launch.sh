dataset_name='zeenea-open-ds'
object_to_annotate='dataset'         # 'column' or 'dataset'
random_state_indexes=(42 48 13 31 88 199 98 3 76 99)         # list of random states for multiple runs


for random_state_index in "${random_state_indexes[@]}"; do
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --generate_semantic_embeddings --enable_semantic_model
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_graph_model
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_binary_classifier_model

    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_hybrid_model_sem_graph_embedding_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_hybrid_model_syn_graph_similarity_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_hybrid_model_syn_sem_similarity_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_hybrid_model_syn_sem_graph_similarity_learning


    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_cross_model_sem_graph_similarity_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_cross_model_syn_sem_similarity_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_cross_model_syn_graph_similarity_learning
    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_cross_model_syn_sem_graph_similarity_learning

    python entrypoint.py --dataset_name=$dataset_name --object_to_annotate=$object_to_annotate --random_state_index=$random_state_index --enable_reciprocal_rank_fusion_model
done

