import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


def load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state):

    files_dir_path = f"{embeddings_dir_path}/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"

    list_name_files = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']
    
    for name_file in list_name_files:
        file_path = f"{files_dir_path}/{name_file}"
        if os.path.isfile(file_path):
            yield torch.load(file_path)
        else:
            print(f'File not found: {file_path}')


def bipartite_negative_sampling_graph(edge_index, num_nodes_b):
    """
    Negative sampling by node in a bipartite graph.
    :param edge_index (Tensor): Positive edges (2 x num_edges) between Set A and Set B.
    :param num_nodes_b (int): Number of nodes in set B.
    :return  (Tensor): Negative sampled edges.
    """
    tuple_list = lambda x,y: (x,y)
    pos_edge_index_tuples = set(map(tuple_list, edge_index[0, :].cpu().numpy().tolist(), edge_index[1, :].cpu().numpy().tolist()))
    neg_edges = []

    for a_node,_ in pos_edge_index_tuples:
        while True:
            # Randomly sample a node from set B
            b_node = torch.randint(num_nodes_b, (1,)).item()
            neg_edge = (a_node, b_node)
            if neg_edge not in pos_edge_index_tuples:
                neg_edges.append(neg_edge)
                break

    return torch.tensor(neg_edges).T


class LinkDataset(torch.utils.data.Dataset):

    def __init__(self, edge_index, src_sem_embed, src_graph_embed, trg_sem_embed, trg_graph_embed, labels):
        self.edge_index = edge_index
        self.src_sem_embed = src_sem_embed
        self.src_graph_embed = src_graph_embed
        self.trg_sem_embed = trg_sem_embed
        self.trg_graph_embed = trg_graph_embed
        self.labels = labels

    def __getitem__(self, item):
        src_id = self.edge_index[0, item]
        trg_id = self.edge_index[1, item]

        return self.src_sem_embed[src_id, :], self.src_graph_embed[src_id, :], self.trg_sem_embed[trg_id, :], self.trg_graph_embed[trg_id, :], self.labels[item]

    def __len__(self):
        return self.edge_index.shape[1]


def collate_fn(batch):

    src_sem_embeddings = torch.stack([x[0] for x in batch])
    src_graph_embeddings = torch.stack([x[1] for x in batch])
    trg_sem_embeddings = torch.stack([x[2] for x in batch])
    trg_graph_embeddings = torch.stack([x[3] for x in batch])
    labels = torch.stack([x[4] for x in batch])

    return src_sem_embeddings, src_graph_embeddings, trg_sem_embeddings, trg_graph_embeddings, labels

class HybridLinkPredictor(torch.nn.Module):

    def __init__(self, semantic_embedding_dim, graph_embedding_dim, hidden_layer_dim, num_classes):
        super().__init__()
        
        self.W_src_g_drop = torch.nn.Linear(in_features=graph_embedding_dim, out_features=hidden_layer_dim)
        self.W_src_w_drop = torch.nn.Linear(in_features=semantic_embedding_dim, out_features=hidden_layer_dim)

        self.W_tgt_g_drop = torch.nn.Linear(in_features=graph_embedding_dim, out_features=hidden_layer_dim)
        self.W_tgt_w_drop = torch.nn.Linear(in_features=semantic_embedding_dim, out_features=hidden_layer_dim)
        
        self.W_src_g_dense = torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim)
        self.W_src_w_dense = torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim)

        self.W_tgt_g_dense = torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim)
        self.W_tgt_w_dense = torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim)
        
        #self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        
        self.dropout = torch.nn.Dropout(p=0.3)
        #self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, src_sem_embed, src_graph_embed, tgt_sem_embed, tgt_graph_embed):

        # apply dropout + feed-forward
        src_g_drop = self.dropout(self.W_src_g_drop(src_graph_embed))
        src_w_drop = self.dropout(self.W_src_w_drop(src_sem_embed))

        tgt_g_drop = self.dropout(self.W_tgt_g_drop(tgt_graph_embed))
        tgt_w_drop = self.dropout(self.W_tgt_w_drop(tgt_sem_embed))

        
        # apply relu + feed-forward 
        src_g_dense = self.relu(self.W_src_g_dense(src_g_drop))
        src_w_dense = self.relu(self.W_src_w_dense(src_w_drop))

        tgt_g_dense = self.tanh(self.W_tgt_g_dense(tgt_g_drop))
        tgt_w_dense = self.tanh(self.W_tgt_w_dense(tgt_w_drop))


        return src_g_dense, src_w_dense, tgt_g_dense, tgt_w_dense


def train_hybrid_model_on_double_cosine_loss(hybrid_link_predictor, optimizer, parameters, positive_edge_loader, negative_edge_loader, device, writer, logger):
    hybrid_link_predictor.train()

    num_classes = parameters['num_classes']
    
    precision_func = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_func = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    f1score_func = MulticlassF1Score( num_classes=num_classes, average=None).to(device)

    cosine_threshold = 0.8
    
    for epoch in range(parameters['nb_epochs']):

        loss_epoch = 0
        loss_batch  = []
        f1score = torch.tensor([]).to(device)
        precision = torch.tensor([]).to(device)
        recall = torch.tensor([]).to(device)
    
        for pos_batch, neg_batch in zip(positive_edge_loader, negative_edge_loader):
    
            pos_src_sem_embed = pos_batch[0]
            pos_src_graph_embed = pos_batch[1]
            pos_trg_sem_embed = pos_batch[2]
            pos_trg_graph_embed = pos_batch[3]
            pos_labels = pos_batch[4]
    
            neg_src_sem_embed = neg_batch[0]
            neg_src_graph_embed = neg_batch[1]
            neg_trg_sem_embed = neg_batch[2]
            neg_trg_graph_embed = neg_batch[3]
            neg_labels = neg_batch[4]
    
            len_batchs = pos_labels.shape[0] + neg_labels.shape[0]
            rand_index = torch.randperm(len_batchs)
    
            src_sem_embed = torch.concat([pos_src_sem_embed, neg_src_sem_embed], dim=0)[rand_index].to(device)
            src_graph_embed = torch.concat([pos_src_graph_embed, neg_src_graph_embed], dim=0)[rand_index].to(device)
            trg_sem_embed = torch.concat([pos_trg_sem_embed, neg_trg_sem_embed], dim=0)[rand_index].to(device)
            trg_graph_embed = torch.concat([pos_trg_graph_embed, neg_trg_graph_embed], dim=0)[rand_index].to(device)
            labels = torch.concat([pos_labels, neg_labels], dim=0)[rand_index].long()
            labels = torch.flatten(labels).to(device)
    
            optimizer.zero_grad()

            src_g_embed, src_w_embed, trg_g_embed, trg_w_embed = hybrid_link_predictor.forward(src_sem_embed, src_graph_embed, trg_sem_embed, trg_graph_embed)
            g_loss = torch.nn.functional.cosine_embedding_loss(src_g_embed, trg_g_embed, labels)
            w_loss = torch.nn.functional.cosine_embedding_loss(src_w_embed, trg_w_embed, labels)
            loss = g_loss + w_loss
            
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())
            # todo: implement early stoping
            
            
            # compute precision recall f1 score on training
            
            with torch.no_grad():
                g_cosine_similarity = torch.nn.functional.cosine_similarity(src_g_embed, trg_g_embed)
                w_cosine_similarity = torch.nn.functional.cosine_similarity(src_w_embed, trg_w_embed)
                cosine_similarity = g_cosine_similarity + w_cosine_similarity
                
                predicitons = torch.where(cosine_similarity >= cosine_threshold, 1, -1)

                binary_labels = torch.where(labels == 1, 1, 0)
                binary_predictions = torch.where(predicitons == 1, 1, 0)
                
                #accuracy = torch.concat([accuracy, accuracy_func(logits, labels).reshape(1, -1)], axis=0)
                f1score = torch.concat([f1score, f1score_func(binary_predictions, binary_labels).reshape(1, -1)], axis=0)
                precision = torch.concat([precision, precision_func(binary_predictions, binary_labels).reshape(1, -1)], axis=0)
                recall = torch.concat([recall, recall_func(binary_predictions, binary_labels).reshape(1, -1)], axis=0)

                
        loss_epoch = sum(loss_batch) / len(loss_batch)
        logger.info(f"Epoch {epoch}, Training Loss: {loss_epoch}")
        logger.info(f"F1-Score: {torch.mean(f1score, dim=0)}, Precision: {torch.mean(precision, dim=0)}, Recall: {torch.mean(recall, dim=0)}")
        
        writer.add_scalar("Train/Loss", loss_epoch, epoch)
        writer.add_scalar("Train/F1Score/NegativeMatch", torch.mean(f1score, dim=0)[0], epoch)
        writer.add_scalar("Train/F1Score/PositiveMatch", torch.mean(f1score, dim=0)[1], epoch)
        writer.add_scalar("Train/Precision/NegativeMatch", torch.mean(precision, dim=0)[0], epoch)
        writer.add_scalar("Train/Precision/PositiveMatch", torch.mean(precision, dim=0)[1], epoch)
        writer.add_scalar("Train/Recall/NegativeMatch", torch.mean(recall, dim=0)[0], epoch)
        writer.add_scalar("Train/Recall/positiveMatch", torch.mean(recall, dim=0)[1], epoch)

    return hybrid_link_predictor

def load_torch_tensor(tensor_dir_path, tensor_name):
    
        return torch.load(f"{tensor_dir_path}/{tensor_name}")


def test_mrr_hits_k_hybrid_double_cosine_sim(
    hybrid_model,
    test_pos_edge_index,
    obj_sem_embeddings,
    obj_graph_embeddings,
    be_sem_embeddings,
    be_graph_embeddings,
    k=10,
    device=None
    ):
    
    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    obj_sem_embeddings = obj_sem_embeddings.to(device)
    obj_graph_embeddings = obj_graph_embeddings.to(device)
    be_sem_embeddings = be_sem_embeddings.to(device)
    be_graph_embeddings = be_graph_embeddings.to(device)
    
    hybrid_model.eval()

    with torch.no_grad():
        
        mrrs = []
        hits_at_k = []

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)

            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]

            # Compute the score for all possible links from src to all target entities
            src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            obj_sem_embed_i = obj_sem_embeddings[src_to_entities_edge_index[0]]
            obj_graph_embed_i = obj_graph_embeddings[src_to_entities_edge_index[0]]    
            be_sem_embed_i = be_sem_embeddings[src_to_entities_edge_index[1]]
            be_graph_embed_i = be_graph_embeddings[src_to_entities_edge_index[1]]
            
            src_g_embed, src_w_embed, trg_g_embed, trg_w_embed = hybrid_model.forward(obj_sem_embed_i, obj_graph_embed_i, be_sem_embed_i, be_graph_embed_i)#[:, 1]
            
            g_cosine_similarity = torch.nn.functional.cosine_similarity(src_g_embed, trg_g_embed)
            w_cosine_similarity = torch.nn.functional.cosine_similarity(src_w_embed, trg_w_embed)
            cosine_similarity = g_cosine_similarity + w_cosine_similarity
            
            # Rank the scores (higher is better), and get the rank of the true edge
            sorted_scores, sorted_indices = torch.sort(cosine_similarity, descending=True)
            
            # get sorted entity ids by score
            sorted_entity_indices = src_to_entities_edge_index[1][sorted_indices]
            #print(f"sorted_entity_indices: {sorted_entity_indices}")

            # get rank of true target entity 
            true_edge_rank = (sorted_entity_indices == tgt).nonzero(as_tuple=True)[0].item()
            
            # MRR Calculation: Reciprocal of the true edge's rank
            mrrs.append(1.0 / (true_edge_rank+1))

            # Hit@K Calculation: Check if the true edge appears in the top K scores
            if true_edge_rank <= k:
                hits_at_k.append(1.0)  # 1 means hit
            else:
                hits_at_k.append(0.0)  # 0 means miss

        # Compute average MRR and Hit@K across all test edges
        mrr = torch.tensor(mrrs).mean().item()
        hit_at_k = torch.tensor(hits_at_k).mean().item()

    return mrr, hit_at_k


def save_metrics(metrics:dict, dataset_name, object_to_predict, random_state, metric_dir):

            if not os.path.exists(metric_dir):
                os.makedirs(metric_dir)
                
            metric_file = open(f"{metric_dir}/link_prediction_{dataset_name}_{object_to_predict}_{random_state}.txt", "w")
            metric_file.write(str(metrics))
            metric_file.close()


def save_model(model, models_dir_path, trained_on_dataset, trained_for_epochs, model_name, random_state):
        model_dir = f"{models_dir_path}/trained_on={trained_on_dataset}/random_state={random_state}/epochs={trained_for_epochs}"
        
        if os.path.exists(model_dir):
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
        else:
            os.makedirs(model_dir)
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")



def main(args):
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Load Arguments")
    dataset_name = args.dataset_name
    object_to_predict = args.object_to_predict
    random_state_index = args.random_state_index

    parameters = {
        "batch_size":args.batch_size,
        "num_workers":args.num_workers,
        "nb_epochs":args.nb_epochs,
        "num_classes":args.num_classes,
        "learning_rate":args.learning_rate,
        'hidden_layer_dim':args.hidden_layer_dim,
        "top_k":args.top_k
    }

    logger.info(args)
    
    
    random_state = [42, 84, 13][random_state_index]

    logger.info("Load Semantic Embeddings")
    embeddings_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings"
    sem_embeddings = list(load_embeddings(embeddings_dir_path, dataset_name, 'semantic-based', random_state))
    col_sem_embeddings = sem_embeddings[0]
    ds_sem_embeddings = sem_embeddings[1]
    be_sem_embeddings = sem_embeddings[2]
    
    logger.info("Load Graph Embeddings")
    graph_embeddings = list(load_embeddings(embeddings_dir_path, dataset_name, 'graph-based', random_state))
    col_graph_embeddings = graph_embeddings[0]
    ds_graph_embeddings = graph_embeddings[1]
    be_graph_embeddings = graph_embeddings[2]

    logger.info("Load Edge Indexes")
    edge_indexes_dir_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/edge_indexes/dataset_name={dataset_name}/random_state={random_state}"
    
    if object_to_predict == 'column':
        train_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_col_edge_index.pt')
        train_neg_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_neg_col_edge_index.pt')
        test_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_col_edge_index.pt')

    elif object_to_predict == 'dataset':
        train_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_ds_edge_index.pt')
        train_neg_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_neg_ds_edge_index.pt')
        test_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_ds_edge_index.pt')
        
    else:
        print("Error in object_to_predict var")

    logger.info("Creates Pos and Neg Labels")
    if object_to_predict == 'column':
        train_pos_col_labels = torch.ones((train_pos_col_edge_index.shape[1], 1))
        train_neg_col_labels = torch.ones((train_neg_col_edge_index.shape[1], 1)) * -1

    elif object_to_predict == 'dataset':
        train_pos_ds_labels = torch.ones((train_pos_ds_edge_index.shape[1], 1))
        train_neg_ds_labels = torch.ones((train_neg_ds_edge_index.shape[1], 1)) * -1

    else:
        print("Error in object_to_predict var")

    
    logger.info("Datasets Creation")
    if object_to_predict == 'column':
        train_pos_edge_dataset = train_pos_edge_dataset = LinkDataset(
                                        train_pos_col_edge_index,
                                        col_sem_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_graph_embeddings,
                                        train_pos_col_labels
                                        )
        
        train_neg_edge_dataset = LinkDataset(
                                        train_neg_col_edge_index,
                                        col_sem_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_graph_embeddings,
                                        train_neg_col_labels
                                        )
        
    elif object_to_predict == 'dataset':
        train_pos_edge_dataset = train_pos_edge_dataset = LinkDataset(
                                        train_pos_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_graph_embeddings,
                                        train_pos_ds_labels
                                        )
        
        train_neg_edge_dataset = LinkDataset(
                                        train_neg_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_graph_embeddings,
                                        train_neg_ds_labels
                                        )
    else:
        print("Error in object_to_predict var")


    logger.info("DataLoaders Creation")
    train_pos_edge_loader = torch.utils.data.DataLoader(train_pos_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])
    train_neg_edge_loader = torch.utils.data.DataLoader(train_neg_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])

    logger.info("Tensorboard SummaryWriter Instatiation")
    writer_log_dir = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/hybrid_model_trainings/dataset_name={dataset_name}/random_state={random_state}/epochs={parameters['nb_epochs']}"

    if not os.path.exists(writer_log_dir):
        os.makedirs(writer_log_dir)
    
    writer = SummaryWriter(writer_log_dir)

    logger.info("Set device to 'cpu' or 'cuda'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Hybrid Model and Optimizer Instantiation")
    if object_to_predict == 'column':
        assert col_sem_embeddings.shape[1] == be_sem_embeddings.shape[1]
        assert col_graph_embeddings.shape[1] == be_graph_embeddings.shape[1]
        
        semantic_embeddings_dim = col_sem_embeddings.shape[1]
        graph_embeddings_dim = col_graph_embeddings.shape[1]

    else:
        assert ds_sem_embeddings.shape[1] == be_sem_embeddings.shape[1]
        assert ds_graph_embeddings.shape[1] == be_graph_embeddings.shape[1]
        
        semantic_embeddings_dim = ds_sem_embeddings.shape[1]
        graph_embeddings_dim = ds_graph_embeddings.shape[1]

    
    hybrid_link_predictor = HybridLinkPredictor(
        semantic_embedding_dim=semantic_embeddings_dim,
        graph_embedding_dim=graph_embeddings_dim,
        hidden_layer_dim=parameters['hidden_layer_dim'],
        num_classes=parameters['num_classes']
        )
    
    hybrid_link_predictor = hybrid_link_predictor.to(device)
    
    optimizer = torch.optim.AdamW(hybrid_link_predictor.parameters(), lr=parameters['learning_rate'])

    logger.info("Hybrid Model Training. Training: Loss, F1Score, Precision, Recall")
    hybrid_link_predictor = train_hybrid_model_on_double_cosine_loss(hybrid_link_predictor, optimizer, parameters, train_pos_edge_loader, train_neg_edge_loader, device, writer, logger)
    writer.flush()
    writer.close()

    logger.info("Test Hybrid Model. Testing: MRR, Hit@10")

    if object_to_predict == 'column':
        mrr, hit_at_10 = test_mrr_hits_k_hybrid_double_cosine_sim(
            hybrid_link_predictor,
            test_pos_col_edge_index, 
            col_sem_embeddings,
            col_graph_embeddings,
            be_sem_embeddings,
            be_graph_embeddings,
            k=parameters['top_k'],
            device=device
            )
    else:
        mrr, hit_at_10 = test_mrr_hits_k_hybrid_double_cosine_sim(
            hybrid_link_predictor,
            test_pos_ds_edge_index, 
            ds_sem_embeddings,
            ds_graph_embeddings,
            be_sem_embeddings,
            be_graph_embeddings,
            k=parameters['top_k'],
            device=device
            )
        

    logger.info(f"MRR: {mrr:.4f}, Hit@10: {hit_at_10:.4f}")

    logger.info("Save metrics")
    
    metric_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/metrics/hybrid-model"
    metrics = {
        "MRR": round(mrr, 4),
        "Hit@10": round(hit_at_10, 4),
        "epochs": parameters["nb_epochs"],
        "random_state":random_state,
        "dataset_name": str(dataset_name)
    }
    
    save_metrics(metrics, dataset_name, object_to_predict, random_state, metric_dir_path)

    logger.info("Save Hybrid Model")
    models_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/models"
    model_name = "HybridLinkPredictor"
    
    save_model(hybrid_link_predictor, models_dir_path, dataset_name, parameters['nb_epochs'], model_name, random_state)







    
    