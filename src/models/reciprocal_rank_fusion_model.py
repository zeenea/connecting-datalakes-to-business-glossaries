import pandas as pd
import logging
import torch
import os
import mlflow
from torch_geometric.data import HeteroData


def compute_rrf():
    pass


def load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state):

    files_dir_path = f"{embeddings_dir_path}/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"

    list_name_files = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']

    for name_file in list_name_files:
        file_path = f"{files_dir_path}/{name_file}"
        if os.path.isfile(file_path):
            yield torch.load(file_path)
        else:
            print(f'File not found: {file_path}')


def load_processed_data(data_dir_path, dataset_name, object_to_annotate, random_state):

    files_dir_path = f"{data_dir_path}/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"

    list_file_names = [
        'train_col_alignments.parquet',
        'test_col_alignments.parquet',
        'train_ds_alignments.parquet',
        'test_ds_alignments.parquet',
        'ds_to_col.parquet',
        'ds_to_be.parquet',
        'be_to_be.parquet'
    ]

    for file_name in list_file_names:

        file_path = f"{files_dir_path}/{file_name}"
        if os.path.isfile(file_path):
            yield pd.read_parquet(file_path)
        else:
            yield pd.DataFrame()


def save_metrics(metrics:dict, dataset_name, object_to_annotate, random_state, metric_dir):

    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    metric_file = open(f"{metric_dir}/link_prediction_{dataset_name}_{object_to_annotate}_{random_state}.txt", "w")
    metric_file.write(str(metrics))
    metric_file.close()


def infer_with_semantic_model(
        test_pos_edge_index,
        obj_sem_embeddings,
        be_sem_embeddings,
        k=10,
        device=None
):

    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    obj_sem_embeddings = obj_sem_embeddings.to(device)
    be_sem_embeddings = be_sem_embeddings.to(device)

    with torch.no_grad():

        sorted_top_k_suggestions = torch.tensor([])

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)

            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]

            # Compute the score for all possible links from src to all target entities
            src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            src_w_embed = obj_sem_embeddings[src_to_entities_edge_index[0]]
            trg_w_embed = be_sem_embeddings[src_to_entities_edge_index[1]]

            cosine_similarity = torch.nn.functional.cosine_similarity(src_w_embed, trg_w_embed)

            # Rank the scores (higher is better), and get the rank of the true edge
            sorted_scores, sorted_indices = torch.sort(cosine_similarity, descending=True)

            # get sorted entity ids by score
            sorted_entity_indices = src_to_entities_edge_index[1][sorted_indices]
            sorted_entity_indices = sorted_entity_indices[:k]
            sorted_top_k_suggestions = torch.concat((sorted_top_k_suggestions, sorted_entity_indices), dim=0)

    return sorted_top_k_suggestions


def create_hetero_graph_dataset(
        object_to_annotate,
        col_embeddings,
        be_embeddings,
        ds_embeddings,
        train_col_alignments,
        test_col_alignments,
        train_ds_alignments,
        test_ds_alignments,
        ds_to_col,
        be_to_be,
        add_col_to_be,
        add_ds_to_col,
        add_ds_to_be,
        add_be_to_be
):
    """Creates Hetero-Graph Dataset."""

    dataset = HeteroData()

    dataset['column'].x = col_embeddings
    dataset['business_entity'].x = be_embeddings
    dataset['dataset'].x = ds_embeddings

    if add_col_to_be:

        train_pos_col_edge_index = torch.from_numpy(train_col_alignments[train_col_alignments['is_matching']==1][['col_id', 'be_id']].values).T
        train_neg_col_edge_index = torch.from_numpy(train_col_alignments[train_col_alignments['is_matching']==0][['col_id', 'be_id']].values).T

        if object_to_annotate == 'column':
            test_pos_col_edge_index =  torch.from_numpy(test_col_alignments[test_col_alignments['is_matching']==1][['col_id', 'be_id']].values).T
            test_neg_col_edge_index =  torch.from_numpy(test_col_alignments[test_col_alignments['is_matching']==0][['col_id', 'be_id']].values).T
        else:
            test_pos_col_edge_index = None
            test_neg_col_edge_index = None

        dataset['column', 'implements', 'business_entity'].edge_index = train_pos_col_edge_index
        dataset['business_entity', 'rev_implements', 'column'].edge_index = torch.flipud(train_pos_col_edge_index)

    if add_ds_to_col:

        ds_to_col_pos_edge_index = torch.from_numpy(ds_to_col.values).T
        dataset['dataset', 'contains', 'column'].edge_index = ds_to_col_pos_edge_index
        dataset['column', 'rev_contains', 'dataset'].edge_index = torch.flipud(ds_to_col_pos_edge_index)

    if add_ds_to_be:

        train_pos_ds_edge_index = torch.from_numpy(train_ds_alignments[train_ds_alignments['is_matching']==1][['ds_id', 'be_id']].values).T
        train_neg_ds_edge_index = torch.from_numpy(train_ds_alignments[train_ds_alignments['is_matching']==0][['ds_id', 'be_id']].values).T

        if object_to_annotate == 'dataset':
            test_pos_ds_edge_index =  torch.from_numpy(test_ds_alignments[test_ds_alignments['is_matching']==1][['ds_id', 'be_id']].values).T
            test_neg_ds_edge_index =  torch.from_numpy(test_ds_alignments[test_ds_alignments['is_matching']==0][['ds_id', 'be_id']].values).T
        else:
            test_pos_ds_edge_index = None
            test_neg_ds_edge_index = None

        dataset['dataset', 'implements', 'business_entity'].edge_index = train_pos_ds_edge_index
        dataset['business_entity', 'rev_implements', 'dataset'].edge_index = torch.flipud(train_pos_ds_edge_index)

    if add_be_to_be:

        be_to_be_pos_edge_index = torch.from_numpy(be_to_be.values).T.type(torch.int64)
        dataset['business_entity', 'composes', 'business_entity'].edge_index = be_to_be_pos_edge_index
        dataset['business_entity', 'rev_composes', 'business_entity'].edge_index = torch.flipud(be_to_be_pos_edge_index)

    return dataset, train_pos_col_edge_index, train_neg_col_edge_index, test_pos_col_edge_index, test_neg_col_edge_index, train_pos_ds_edge_index, train_neg_ds_edge_index, test_pos_ds_edge_index, test_neg_ds_edge_index, ds_to_col_pos_edge_index, be_to_be_pos_edge_index


def infer_with_graph_model(col_embeddings, ds_embeddings, be_embeddings, source_object, hetero_model, train_pos_edge_index, test_pos_edge_index, k=10, device=None):
    hetero_model.eval()

    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    with torch.no_grad():
        # Encode node embeddings using the trained GraphSAGE model
        x_dict = {}
        x_dict['column'] = col_embeddings.to(device)
        x_dict['dataset'] = ds_embeddings.to(device)
        x_dict['business_entity'] = be_embeddings.to(device)

        edge_index_dict = {}

        z = hetero_model.encode(x_dict, train_pos_edge_index)

        sorted_top_k_suggestions = torch.tensor([])

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)
            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]

            # Compute the score for all possible links from src to all target entities
            all_entity_ids = all_entity_ids.reshape(1, -1)
            tensor_src = src.repeat(all_entity_ids.shape[1]).reshape(1, -1)

            assert tensor_src.shape[1] == all_entity_ids.shape[1]

            src_to_entities_edge_index = torch.concat([tensor_src, all_entity_ids], dim=0).to(device)
            target_true_index = (src_to_entities_edge_index[1,:] == tgt).nonzero(as_tuple=True)[0].to(device)

            embeddings1 = z[source_object][src_to_entities_edge_index[0]]
            embeddings2 = z['business_entity'][src_to_entities_edge_index[1]]

            # cosine similarity
            edge_scores = hetero_model.decode(embeddings1, embeddings2)

            # Rank the scores (higher is better), and get the rank of the true edge
            sorted_scores, sorted_indices = torch.sort(edge_scores, descending=True)

            sorted_entity_indices = src_to_entities_edge_index[1][sorted_indices]
            sorted_entity_indices = sorted_entity_indices[:k]
            sorted_top_k_suggestions = torch.concat((sorted_top_k_suggestions, sorted_entity_indices), dim=0)

    return sorted_top_k_suggestions


def load_torch_tensor(tensor_dir_path, tensor_name):

    return torch.load(f"{tensor_dir_path}/{tensor_name}")


def create_dataset_edge_index(object_to_annotate,
                                ds_to_col_pos_edge_index,
                                be_to_be_pos_edge_index,
                                train_pos_col_edge_index,
                                test_pos_col_edge_index,
                                train_pos_ds_edge_index,
                                test_pos_ds_edge_index,
                                device):
    pos_edge_index_dict = {}

    pos_edge_index_dict['dataset', 'contains', 'column'] = ds_to_col_pos_edge_index.to(device)
    pos_edge_index_dict['column', 'rev_contains', 'dataset'] = torch.flipud(ds_to_col_pos_edge_index).to(device)

    pos_edge_index_dict['business_entity', 'composes', 'business_entity'] = be_to_be_pos_edge_index.to(device)
    pos_edge_index_dict['business_entity', 'rev_composes', 'business_entity'] = torch.flipud(be_to_be_pos_edge_index).to(device)

    if object_to_annotate == 'column':

        pos_edge_index_dict['column', 'implements', 'business_entity'] = test_pos_col_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_implements', 'column'] = torch.flipud(test_pos_col_edge_index).to(device)

        #pos_ds_edge_index = torch.concat((train_pos_ds_edge_index, test_pos_ds_edge_index), dim=1)
        pos_edge_index_dict['dataset', 'implements', 'business_entity'] = train_pos_ds_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(train_pos_ds_edge_index).to(device)

    if object_to_annotate == 'dataset':

        pos_edge_index_dict['dataset', 'implements', 'business_entity'] = test_pos_ds_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(test_pos_ds_edge_index).to(device)

        #pos_col_edge_index = torch.concat((train_pos_col_edge_index, test_pos_col_edge_index), dim=1)
        pos_edge_index_dict['column', 'implements', 'business_entity'] = train_pos_col_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_implements', 'column'] = torch.flipud(train_pos_col_edge_index).to(device)

    return pos_edge_index_dict


def load_processed_data(data_dir_path, dataset_name, object_to_annotate, random_state):

    files_dir_path = f"{data_dir_path}/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"

    list_file_names = [
        'train_col_alignments.parquet',
        'test_col_alignments.parquet',
        'train_ds_alignments.parquet',
        'test_ds_alignments.parquet',
        'ds_to_col.parquet',
        'ds_to_be.parquet',
        'be_to_be.parquet'
    ]

    for file_name in list_file_names:

        file_path = f"{files_dir_path}/{file_name}"
        if os.path.isfile(file_path):
            yield pd.read_parquet(file_path)
        else:
            yield pd.DataFrame()


def main(args):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Load Arguments")

    dataset_name = args.dataset_name
    object_to_annotate = args.object_to_annotate
    random_state_index = args.random_state_index
    parameters = {
        "top_k":args.top_k,
        "nb_epochs":0}

    logger.info(args)

    random_state = [42, 84, 13][random_state_index]

    logger.info('Load semantic embeddings')
    model_type = "semantic-based"
    embeddings_dir_path = "../gold_data/embeddings"
    embeddings_out = list(load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state))
    col_sem_embeddings = embeddings_out[0]
    ds_sem_embeddings = embeddings_out[1]
    be_sem_embeddings = embeddings_out[2]

    assert col_sem_embeddings.shape[1] == ds_sem_embeddings.shape[1]
    assert col_sem_embeddings.shape[1] == be_sem_embeddings.shape[1]

    logger.info('Load graph embeddings')
    model_type = "graph-based"
    embeddings_dir_path = "../gold_data/embeddings"
    embeddings_out = list(load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state))
    col_graph_embeddings = embeddings_out[0]
    ds_graph_embeddings = embeddings_out[1]
    be_graph_embeddings = embeddings_out[2]

    assert col_graph_embeddings.shape[1] == ds_graph_embeddings.shape[1]
    assert col_graph_embeddings.shape[1] == be_graph_embeddings.shape[1]

    logger.info('Load processed data')
    data_dir_path = "../gold_data/raw_to_dataframes"
    data_out = list(load_processed_data(data_dir_path, dataset_name, object_to_annotate, random_state))
    train_col_alignments = data_out[0]
    test_col_alignments = data_out[1]
    train_ds_alignments = data_out[2]
    test_ds_alignments = data_out[3]
    ds_to_col = data_out[4]
    ds_to_be = data_out[5]
    be_to_be = data_out[6]

    logger.info("Load Edge Indexes")
    edge_indexes_dir_path = f"../gold_data/edge_indexes/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"

    if object_to_annotate == 'column':
        train_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_col_edge_index.pt')
        test_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_col_edge_index.pt')

        train_pos_ds_edge_index = torch.from_numpy(train_ds_alignments[train_ds_alignments['is_matching'] == 1][['ds_id', 'be_id']].values).T
        test_pos_ds_edge_index = None

    elif object_to_annotate == 'dataset':
        train_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_ds_edge_index.pt')
        test_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_ds_edge_index.pt')

        train_pos_col_edge_index = torch.from_numpy(train_col_alignments[train_col_alignments['is_matching'] == 1][['col_id', 'be_id']].values).T
        test_pos_col_edge_index = None

    ds_to_col_pos_edge_index = load_torch_tensor(edge_indexes_dir_path, 'ds_to_col_pos_edge_index.pt')
    be_to_be_pos_edge_index = load_torch_tensor(edge_indexes_dir_path, 'be_to_be_pos_edge_index.pt')

    print(ds_to_col_pos_edge_index.shape)
    print(be_to_be_pos_edge_index.shape)
    print(train_pos_ds_edge_index.shape)
    print(test_pos_ds_edge_index.shape)
    print(train_pos_col_edge_index.shape)

    logger.info("Set device to 'cpu' or 'cuda'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Create dataset edge index")

    dataset_edge_index = create_dataset_edge_index(object_to_annotate=object_to_annotate,
                                                   ds_to_col_pos_edge_index=ds_to_col_pos_edge_index,
                                                   be_to_be_pos_edge_index=be_to_be_pos_edge_index,
                                                   train_pos_col_edge_index=train_pos_col_edge_index,
                                                   test_pos_col_edge_index=test_pos_col_edge_index,
                                                   train_pos_ds_edge_index=train_pos_ds_edge_index,
                                                   test_pos_ds_edge_index=test_pos_ds_edge_index,
                                                   device=device
                                                   )


    logger.info("Load Graph Model")

    model_class_name = "HeteroGraphSage"
    registered_model_name = f"{dataset_name}-{random_state_index}-{object_to_annotate}-{model_class_name}"
    graph_model = mlflow.pytorch.load_model(f"models:/{registered_model_name}/Latest")
    graph_model.to(device)

    logger.info("MLFlow managing")
    mlflow.set_experiment('reciprocal_rank_fusion_model')

    with mlflow.start_run():

        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag('object_to_annotate', object_to_annotate)
        mlflow.log_params(parameters)
        mlflow.log_param('dataset_split_random_state', random_state)
        mlflow.log_param('language_model_name', 'all-MiniLM-L6-v2')
        mlflow.log_param('semantic_embedding_dimension', col_sem_embeddings.shape[1])
        mlflow.log_param('graph_embedding_dimension', col_graph_embeddings.shape[1])

        # inference

        logger.info("Infer with Semantic Model")

        if object_to_annotate == 'column':

            #test_pos_col_edge_index = torch.from_numpy(test_col_alignments[test_col_alignments['is_matching']==1][['col_id', 'be_id']].values).T

            semantic_top_k_suggestions = infer_with_semantic_model(
                test_pos_col_edge_index,
                col_sem_embeddings,
                be_sem_embeddings,
                k=parameters["top_k"],
                device=device
            )

            graph_top_k_suggestions = infer_with_graph_model(
                col_embeddings=col_graph_embeddings,
                ds_embeddings=ds_graph_embeddings,
                be_embeddings=be_graph_embeddings,
                source_object=object_to_annotate,
                hetero_model=graph_model,
                train_pos_edge_index=dataset_edge_index,
                test_pos_edge_index=test_pos_col_edge_index,
                k=parameters['top_k'],
                device=device
            )
        else:
            #test_pos_ds_edge_index = torch.from_numpy(test_ds_alignments[test_ds_alignments['is_matching']==1][['ds_id', 'be_id']].values).T

            semantic_top_k_suggestions = infer_with_semantic_model(
                test_pos_ds_edge_index,
                ds_sem_embeddings,
                be_sem_embeddings,
                k=parameters["top_k"],
                device=device
            )

            print(type(train_pos_ds_edge_index))
            print(train_pos_ds_edge_index.shape)

            graph_top_k_suggestions = infer_with_graph_model(
                col_embeddings=col_graph_embeddings,
                ds_embeddings=ds_graph_embeddings,
                be_embeddings=be_graph_embeddings,
                source_object=object_to_annotate,
                hetero_model=graph_model,
                train_pos_edge_index=dataset_edge_index,
                test_pos_edge_index=test_pos_ds_edge_index,
                k=parameters['top_k'],
                device=device
            )

        logger.info("Compute final ranking with RRF")

        print(semantic_top_k_suggestions.shape)
        print(graph_top_k_suggestions.shape)

        logger.info("Compute MRR and Hit@K")

        mrr = 0
        hit_at_k = 0

        logger.info("Save metrics")

        metric_dir_path = "../gold_data/metrics/semantic-model"
        metrics = {
            "MRR": round(mrr, 4),
            f"Hit@{parameters['top_k']}": round(hit_at_k, 4),
            "random_state": random_state,
            "dataset_name": str(dataset_name)
        }

        save_metrics(metrics, dataset_name, object_to_annotate, random_state, metric_dir_path)

        # compute rrf

        # compute mrr and hit@10

        mlflow.log_metric('mrr', round(mrr, 4))
        mlflow.log_metric(f"hit_at_{parameters['top_k']}", round(hit_at_k, 4))





if __name__ == "__main__":

    main()
