import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import mlflow
import yaml
import argparse

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ['TORCH_USE_CUDA_DSA']='1'

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
        
def load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state):

    files_dir_path = f"{embeddings_dir_path}/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"

    list_name_files = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']
    
    for name_file in list_name_files:
        file_path = f"{files_dir_path}/{name_file}"
        if os.path.isfile(file_path):
            yield torch.load(file_path)
        else:
            print(f'File not found: {file_path}')



class LinkDataset(torch.utils.data.Dataset):

    def __init__(self, edge_index, src_sem_embed, src_syn_embed, src_graph_embed, trg_sem_embed, trg_syn_embed, trg_graph_embed, labels):
        self.edge_index = edge_index
        self.src_sem_embed = src_sem_embed
        self.src_syn_embed = src_syn_embed
        self.src_graph_embed = src_graph_embed
        self.trg_sem_embed = trg_sem_embed
        self.trg_syn_embed = trg_syn_embed
        self.trg_graph_embed = trg_graph_embed
        self.labels = labels

    def __getitem__(self, item):
        src_id = self.edge_index[0, item]
        trg_id = self.edge_index[1, item]
        
        src_sem_embed = self.src_sem_embed[src_id, :]
        src_syn_embed = self.src_syn_embed[src_id, :]
        src_graph_embed = self.src_graph_embed[src_id, :]
        trg_sem_embed = self.trg_sem_embed[trg_id, :]
        trg_syn_embed = self.trg_syn_embed[trg_id, :]
        trg_graph_embed = self.trg_graph_embed[trg_id, :]
        label = self.labels[item]

        return src_sem_embed, src_syn_embed, src_graph_embed, trg_sem_embed, trg_syn_embed, trg_graph_embed, label

    def __len__(self):
        return self.edge_index.shape[1]


def collate_fn(batch):

    src_sem_embeddings = torch.stack([x[0] for x in batch])
    src_syn_embeddings = torch.stack([x[1] for x in batch])
    src_graph_embeddings = torch.stack([x[2] for x in batch])
    trg_sem_embeddings = torch.stack([x[3] for x in batch])
    trg_syn_embeddings = torch.stack([x[4] for x in batch])
    trg_graph_embeddings = torch.stack([x[5] for x in batch])
    labels = torch.stack([x[6] for x in batch])

    return src_sem_embeddings, src_syn_embeddings, src_graph_embeddings, trg_sem_embeddings, trg_syn_embeddings, trg_graph_embeddings, labels


def train_xgboost_model(grid_search, positive_edge_loader, negative_edge_loader, device, logger):
    

    all_cosine_sim = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    
    for pos_batch, neg_batch in zip(positive_edge_loader, negative_edge_loader):

        pos_src_sem_embed = pos_batch[0]
        pos_src_syn_embed = pos_batch[1]
        pos_src_graph_embed = pos_batch[2]
        pos_tgt_sem_embed = pos_batch[3]
        pos_tgt_syn_embed = pos_batch[4]
        pos_tgt_graph_embed = pos_batch[5]
        pos_labels = pos_batch[6]

        neg_src_sem_embed = neg_batch[0]
        neg_src_syn_embed = neg_batch[1]
        neg_src_graph_embed = neg_batch[2]
        neg_tgt_sem_embed = neg_batch[3]
        neg_tgt_syn_embed = neg_batch[4]
        neg_tgt_graph_embed = neg_batch[5]
        neg_labels = neg_batch[6]

        len_batchs = pos_labels.shape[0] + neg_labels.shape[0]
        rand_index = torch.randperm(len_batchs)

        src_sem_embed = torch.concat([pos_src_sem_embed, neg_src_sem_embed], dim=0)[rand_index].to(device)
        src_syn_embed = torch.concat([pos_src_syn_embed, neg_src_syn_embed], dim=0)[rand_index].to(device)
        src_graph_embed = torch.concat([pos_src_graph_embed, neg_src_graph_embed], dim=0)[rand_index].to(device)
        
        tgt_sem_embed = torch.concat([pos_tgt_sem_embed, neg_tgt_sem_embed], dim=0)[rand_index].to(device)
        tgt_syn_embed = torch.concat([pos_tgt_syn_embed, neg_tgt_syn_embed], dim=0)[rand_index].to(device)
        tgt_graph_embed = torch.concat([pos_tgt_graph_embed, neg_tgt_graph_embed], dim=0)[rand_index].to(device)

        cosine_sim_sem = torch.nn.functional.cosine_similarity(src_sem_embed, tgt_sem_embed)
        cosine_sim_sem = cosine_sim_sem.reshape(-1, 1)
        
        # cosine similarity on syntactic embeddings
        cosine_sim_syn = torch.nn.functional.cosine_similarity(src_syn_embed, tgt_syn_embed)
        cosine_sim_syn = cosine_sim_syn.reshape(-1, 1)

        # cosine similarity on graph embeddings
        cosine_sim_graph = torch.nn.functional.cosine_similarity(src_graph_embed, tgt_graph_embed)
        cosine_sim_graph = cosine_sim_graph.reshape(-1, 1)        
        
        # matrix of cosine similarities with a size of (batch_size x 2)
        cosine_sim_matrix = torch.concat([cosine_sim_sem, cosine_sim_syn, cosine_sim_graph], dim=1)

        
        labels = torch.concat([pos_labels, neg_labels], dim=0)[rand_index].long()
        labels = torch.flatten(labels).to(device)
        labels = labels.reshape(-1, 1).float()

        all_cosine_sim = torch.concat([all_cosine_sim, cosine_sim_matrix], dim=0)
        all_labels = torch.concat([all_labels, labels], dim=0)

    all_cosine_sim = all_cosine_sim.detach().cpu()
    all_labels = all_labels.detach().cpu()

    # grid search cv to get best estimator
    grid_search.fit(all_cosine_sim, all_labels)
    link_predictor_model = grid_search.best_estimator_
    #link_predictor_model.fit(all_cosine_sim, all_labels)

    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    # compute accuracy, precision, recall, and f1-score
    predictions = link_predictor_model.predict(all_cosine_sim)
    
    accuracy = accuracy_score(predictions, all_labels)
    f1score = f1_score(predictions, all_labels, average=None)
    precision = precision_score(predictions, all_labels, average=None)
    recall = recall_score(predictions, all_labels, average=None)
            
    logger.info(f"Train Accuracy: {accuracy}")
    logger.info(f"F1-Score: {f1score}, Precision: {precision}, Recall: {recall}")
    
    metrics = {
        "Train/Accuracy": accuracy,
        "Train/F1Score/NegativeMatch": f1score[0], 
        "Train/F1Score/PositiveMatch":f1score[1], 
        "Train/Precision/NegativeMatch": precision[0], 
        "Train/Precision/PositiveMatch": precision[1], 
        "Train/Recall/NegativeMatch": recall[0], 
        "Train/Recall/positiveMatch": recall[1]
    }
    mlflow.log_metrics(metrics)

    return link_predictor_model

def test_xgboost_model(link_predictor_model, positive_edge_loader, negative_edge_loader, device, logger):
    
    all_cosine_sim = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    
    for pos_batch, neg_batch in zip(positive_edge_loader, negative_edge_loader):

        pos_src_sem_embed = pos_batch[0]
        pos_src_syn_embed = pos_batch[1]
        pos_src_graph_embed = pos_batch[2]
        pos_tgt_sem_embed = pos_batch[3]
        pos_tgt_syn_embed = pos_batch[4]
        pos_tgt_graph_embed = pos_batch[5]
        pos_labels = pos_batch[6]

        neg_src_sem_embed = neg_batch[0]
        neg_src_syn_embed = neg_batch[1]
        neg_src_graph_embed = neg_batch[2]
        neg_tgt_sem_embed = neg_batch[3]
        neg_tgt_syn_embed = neg_batch[4]
        neg_tgt_graph_embed = neg_batch[5]
        neg_labels = neg_batch[6]

        len_batchs = pos_labels.shape[0] + neg_labels.shape[0]
        rand_index = torch.randperm(len_batchs)

        src_sem_embed = torch.concat([pos_src_sem_embed, neg_src_sem_embed], dim=0)[rand_index].to(device)
        src_syn_embed = torch.concat([pos_src_syn_embed, neg_src_syn_embed], dim=0)[rand_index].to(device)
        src_graph_embed = torch.concat([pos_src_graph_embed, neg_src_graph_embed], dim=0)[rand_index].to(device)
        
        tgt_sem_embed = torch.concat([pos_tgt_sem_embed, neg_tgt_sem_embed], dim=0)[rand_index].to(device)
        tgt_syn_embed = torch.concat([pos_tgt_syn_embed, neg_tgt_syn_embed], dim=0)[rand_index].to(device)
        tgt_graph_embed = torch.concat([pos_tgt_graph_embed, neg_tgt_graph_embed], dim=0)[rand_index].to(device)

        cosine_sim_sem = torch.nn.functional.cosine_similarity(src_sem_embed, tgt_sem_embed)
        cosine_sim_sem = cosine_sim_sem.reshape(-1, 1)
        
        # cosine similarity on syntactic embeddings
        cosine_sim_syn = torch.nn.functional.cosine_similarity(src_syn_embed, tgt_syn_embed)
        cosine_sim_syn = cosine_sim_syn.reshape(-1, 1)

        # cosine similarity on graph embeddings
        cosine_sim_graph = torch.nn.functional.cosine_similarity(src_graph_embed, tgt_graph_embed)
        cosine_sim_graph = cosine_sim_graph.reshape(-1, 1)        
        
        # matrix of cosine similarities with a size of (batch_size x 2)
        cosine_sim_matrix = torch.concat([cosine_sim_sem, cosine_sim_syn, cosine_sim_graph], dim=1)

        
        labels = torch.concat([pos_labels, neg_labels], dim=0)[rand_index].long()
        labels = torch.flatten(labels).to(device)
        labels = labels.reshape(-1, 1).float()

        all_cosine_sim = torch.concat([all_cosine_sim, cosine_sim_matrix], dim=0)
        all_labels = torch.concat([all_labels, labels], dim=0)

    all_cosine_sim = all_cosine_sim.detach().cpu()
    all_labels = all_labels.detach().cpu()

    # compute accuracy, precision, recall, and f1-score
    predictions = link_predictor_model.predict(all_cosine_sim)
    accuracy = accuracy_score(predictions, all_labels)
    f1score = f1_score(predictions, all_labels, average=None)
    precision = precision_score(predictions, all_labels, average=None)
    recall = recall_score(predictions, all_labels, average=None)
            
    logger.info(f"Test Accuracy: {accuracy}")
    logger.info(f"F1-Score: {f1score}, Precision: {precision}, Recall: {recall}")
    
    metrics = {
        "Test/Accuracy": accuracy,
        "Test/F1Score/NegativeMatch": f1score[0], 
        "Test/F1Score/PositiveMatch":f1score[1], 
        "Test/Precision/NegativeMatch": precision[0], 
        "Test/Precision/PositiveMatch": precision[1], 
        "Test/Recall/NegativeMatch": recall[0], 
        "Test/Recall/positiveMatch": recall[1]
    }
    mlflow.log_metrics(metrics)
    

def load_torch_tensor(tensor_dir_path, tensor_name):
    
        return torch.load(f"{tensor_dir_path}/{tensor_name}")


def test_mrr_hits_k(
    link_predictor_model,
    test_pos_edge_index,
    obj_sem_embeddings,
    obj_syn_embeddings,
    obj_graph_embeddings,
    be_sem_embeddings,
    be_syn_embeddings,
    be_graph_embeddings,
    k=10,
    device=None
    ):
    
    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    obj_sem_embeddings = obj_sem_embeddings.to(device)
    obj_syn_embeddings = obj_syn_embeddings.to(device)
    obj_graph_embeddings = obj_graph_embeddings.to(device)
    be_sem_embeddings = be_sem_embeddings.to(device)
    be_syn_embeddings = be_syn_embeddings.to(device)
    be_graph_embeddings = be_graph_embeddings.to(device)
    
    #link_predictor_model.eval()

    with torch.no_grad():
        
        mrrs = []
        hits_at_k = []

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)

            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]
        
            # Compute the score for all possible links from src to all target entities
            src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            
            src_sem_embed = obj_sem_embeddings[src_to_entities_edge_index[0]]
            src_syn_embed = obj_syn_embeddings[src_to_entities_edge_index[0]]  
            src_graph_embed = obj_graph_embeddings[src_to_entities_edge_index[0]]
            
            tgt_sem_embed = be_sem_embeddings[src_to_entities_edge_index[1]]
            tgt_syn_embed = be_syn_embeddings[src_to_entities_edge_index[1]]
            tgt_graph_embed = be_graph_embeddings[src_to_entities_edge_index[1]]
        
            cosine_sim_sem = torch.nn.functional.cosine_similarity(src_sem_embed, tgt_sem_embed)
            cosine_sim_sem = cosine_sim_sem.reshape(-1, 1)
            
            # cosine similarity on syntactic embeddings
            cosine_sim_syn = torch.nn.functional.cosine_similarity(src_syn_embed, tgt_syn_embed)
            cosine_sim_syn = cosine_sim_syn.reshape(-1, 1)
        
            # cosine similarity on graph embeddings
            cosine_sim_graph = torch.nn.functional.cosine_similarity(src_graph_embed, tgt_graph_embed)
            cosine_sim_graph = cosine_sim_graph.reshape(-1, 1)        
            
            # matrix of cosine similarities with a size of (batch_size x 2)
            cosine_sim_matrix = torch.concat([cosine_sim_sem, cosine_sim_syn, cosine_sim_graph], dim=1)
            cosine_sim_matrix = cosine_sim_matrix.detach().cpu()
            
            preds_proba = link_predictor_model.predict_proba(cosine_sim_matrix)
            preds_proba = preds_proba[:, 1]
            preds_proba = torch.tensor(preds_proba)

            # Rank the scores (higher is better), and get the rank of the true edge
            sorted_scores, sorted_indices = torch.sort(torch.flatten(preds_proba), descending=True)
            
            # get sorted entity ids by score
            sorted_entity_indices = src_to_entities_edge_index[1][sorted_indices]

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
    

def save_metrics(metrics:dict, dataset_name, object_to_annotate, random_state, metric_dir):

            if not os.path.exists(metric_dir):
                os.makedirs(metric_dir)
                
            metric_file = open(f"{metric_dir}/link_prediction_{dataset_name}_{object_to_annotate}_{random_state}.txt", "w")
            metric_file.write(str(metrics))
            metric_file.close()


def save_model(model, models_dir_path, trained_on_dataset, object_to_annotate, trained_for_epochs, model_name, random_state):
        model_dir = f"{models_dir_path}/model_name={model_name}/trained_on={trained_on_dataset}/object_to_annotate={object_to_annotate}/random_state={random_state}/epochs={trained_for_epochs}"
        
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
    object_to_annotate = args.object_to_annotate
    random_state_index = args.random_state_index

    
    parameters = {
        "batch_size":args.batch_size,
        "nb_epochs":args.nb_epochs,
        "num_workers":args.num_workers,
        "num_classes":args.num_classes,
        "top_k":args.top_k
    }
    
    
    logger.info(args)

    logger.info("Set device to 'cpu' or 'cuda'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    random_state = [42, 84, 13][random_state_index]

    logger.info("Load Semantic Embeddings")
    embeddings_dir_path = "../gold_data/embeddings"
    sem_embeddings = list(load_embeddings(embeddings_dir_path, dataset_name, 'semantic-based', random_state))
    col_sem_embeddings = sem_embeddings[0].float()
    ds_sem_embeddings = sem_embeddings[1].float()
    be_sem_embeddings = sem_embeddings[2].float()
    
    logger.info("Load Syntactic Embeddings")
    embeddings_dir_path = "../gold_data/embeddings"
    syn_embeddings = list(load_embeddings(embeddings_dir_path, dataset_name, 'syntactic-based', random_state))
    col_syn_embeddings = syn_embeddings[0].float()
    ds_syn_embeddings = syn_embeddings[1].float()
    be_syn_embeddings = syn_embeddings[2].float()

    logger.info("load Graph Embeddings")
    embeddings_dir_path = "../gold_data/embeddings"
    graph_embeddings = list(load_embeddings(embeddings_dir_path, dataset_name, 'graph-based', random_state))
    col_graph_embeddings = graph_embeddings[0].detach().cpu().float()
    ds_graph_embeddings = graph_embeddings[1].detach().cpu().float()
    be_graph_embeddings = graph_embeddings[2].detach().cpu().float()
    

    logger.info("Load Edge Indexes")
    edge_indexes_dir_path = f"../gold_data/edge_indexes/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"
    
    if object_to_annotate == 'column':
        train_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_col_edge_index.pt')
        train_neg_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_neg_col_edge_index.pt')
        test_pos_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_col_edge_index.pt')
        test_neg_col_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_neg_col_edge_index.pt')

    elif object_to_annotate == 'dataset':
        train_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_pos_ds_edge_index.pt')
        train_neg_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'train_neg_ds_edge_index.pt')
        test_pos_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_pos_ds_edge_index.pt')
        test_neg_ds_edge_index = load_torch_tensor(edge_indexes_dir_path, 'test_neg_ds_edge_index.pt')
        
    else:
        print("Error in object_to_annotate var")

    logger.info("Creates Pos and Neg Labels")
    if object_to_annotate == 'column':
        train_pos_col_labels = torch.ones((train_pos_col_edge_index.shape[1], 1))
        train_neg_col_labels = torch.zeros((train_neg_col_edge_index.shape[1], 1))

        test_pos_col_labels = torch.ones((test_pos_col_edge_index.shape[1], 1))
        test_neg_col_labels = torch.zeros((test_neg_col_edge_index.shape[1], 1))


    elif object_to_annotate == 'dataset':
        train_pos_ds_labels = torch.ones((train_pos_ds_edge_index.shape[1], 1))
        train_neg_ds_labels = torch.zeros((train_neg_ds_edge_index.shape[1], 1))

        test_pos_ds_labels = torch.ones((test_pos_ds_edge_index.shape[1], 1))
        test_neg_ds_labels = torch.zeros((test_neg_ds_edge_index.shape[1], 1))


    else:
        print("Error in object_to_annotate var")

    
    logger.info("Datasets Creation")
    if object_to_annotate == 'column':
        train_pos_edge_dataset = LinkDataset(
                                        train_pos_col_edge_index,
                                        col_sem_embeddings,
                                        col_syn_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        train_pos_col_labels
                                        )
        
        train_neg_edge_dataset = LinkDataset(
                                        train_neg_col_edge_index,
                                        col_sem_embeddings,
                                        col_syn_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        train_neg_col_labels
                                        )
        test_pos_edge_dataset = LinkDataset(
                                        test_pos_col_edge_index,
                                        col_sem_embeddings,
                                        col_syn_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        test_pos_col_labels
                                        )
        
        test_neg_edge_dataset = LinkDataset(
                                        test_neg_col_edge_index,
                                        col_sem_embeddings,
                                        col_syn_embeddings,
                                        col_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        test_neg_col_labels
                                        )
        
    elif object_to_annotate == 'dataset':
        train_pos_edge_dataset = LinkDataset(
                                        train_pos_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_syn_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        train_pos_ds_labels
                                        )
        
        train_neg_edge_dataset = LinkDataset(
                                        train_neg_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_syn_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        train_neg_ds_labels
                                        )

        test_pos_edge_dataset = LinkDataset(
                                        test_pos_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_syn_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        test_pos_ds_labels
                                        )
        
        test_neg_edge_dataset = LinkDataset(
                                        test_neg_ds_edge_index,
                                        ds_sem_embeddings,
                                        ds_syn_embeddings,
                                        ds_graph_embeddings,
                                        be_sem_embeddings,
                                        be_syn_embeddings,
                                        be_graph_embeddings,
                                        test_neg_ds_labels
                                        )
    else:
        print("Error in object_to_annotate var")

    
    logger.info("DataLoaders Creation")
    train_pos_edge_loader = torch.utils.data.DataLoader(train_pos_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])
    train_neg_edge_loader = torch.utils.data.DataLoader(train_neg_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])

    test_pos_edge_loader = torch.utils.data.DataLoader(test_pos_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])
    test_neg_edge_loader = torch.utils.data.DataLoader(test_neg_edge_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])


    logger.info("Cross Model and Optimizer Instantiation")
    if object_to_annotate == 'column':
        assert col_sem_embeddings.shape[1] == be_sem_embeddings.shape[1]
        assert col_syn_embeddings.shape[1] == be_syn_embeddings.shape[1]
        assert col_graph_embeddings.shape[1] == be_graph_embeddings.shape[1]
        
        semantic_embeddings_dim = col_sem_embeddings.shape[1]
        syntactic_embeddings_dim = col_syn_embeddings.shape[1]
        graph_embeddings_dim = col_graph_embeddings.shape[1]

    else:
        assert ds_sem_embeddings.shape[1] == be_sem_embeddings.shape[1]
        assert ds_syn_embeddings.shape[1] == be_syn_embeddings.shape[1]
        assert ds_graph_embeddings.shape[1] == be_graph_embeddings.shape[1]
        
        semantic_embeddings_dim = ds_sem_embeddings.shape[1]
        syntactic_embeddings_dim = ds_syn_embeddings.shape[1]
        graph_embeddings_dim = ds_graph_embeddings.shape[1]

    link_predictor_model = XGBClassifier()
    model_class_name = "XGBoostClassifier"
    
    param_grid = {
            'eta':[0.3, 0.2, 0.1, 0.01, 0.001],
            'max_depth':[5, 10, 20, 30, None],
            'subsample':[0.5, 0.7, 1],
            'eval_metric':['ndcg', 'auc', 'logloss', 'map'],
            'objective':['binary:logistic', 'binary:hinge', 'rank:ndcg']
        }

    grid_search = GridSearchCV(
        estimator=link_predictor_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=3,
        scoring='balanced_accuracy',
        refit = True
    )
    

    logger.info("MLFlow managing")
    mlflow.set_experiment('xgboost_classifier_syntactic_semantic_graph_similarity_model')
    
    with mlflow.start_run():
        
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag('object_to_annotate', object_to_annotate)
        mlflow.log_param('dataset_split_random_state', random_state)
        mlflow.log_param('link_predictor_model', model_class_name)    
        mlflow.log_params(parameters)

        logger.info("XGBoostClassifier. Training: Accuracy, F1Score, Precision, Recall")
        
        link_predictor_model = train_xgboost_model(grid_search, train_pos_edge_loader, train_neg_edge_loader, device, logger)

        logger.info("Testing: Accuracy, F1Score, Precision, Recall")
        test_xgboost_model(link_predictor_model, test_pos_edge_loader, test_neg_edge_loader, device, logger)

        logger.info("Testing: MRR, Hit@10")

        if object_to_annotate == 'column':
            mrr, hit_at_10 = test_mrr_hits_k(
                link_predictor_model,
                test_pos_col_edge_index, 
                col_sem_embeddings,
                col_syn_embeddings,
                col_graph_embeddings,
                be_sem_embeddings,
                be_syn_embeddings,
                be_graph_embeddings,
                k=parameters['top_k'],
                device=device
                )
        else:
            mrr, hit_at_10 = test_mrr_hits_k(
                link_predictor_model,
                test_pos_ds_edge_index, 
                ds_sem_embeddings,
                ds_syn_embeddings,
                ds_graph_embeddings,
                be_sem_embeddings,
                be_syn_embeddings,
                be_graph_embeddings,
                k=parameters['top_k'],
                device=device
                )
            
        
        logger.info(f"MRR: {mrr:.4f}, Hit@10: {hit_at_10:.4f}")
        
            
        logger.info("Save metrics")
        
        metric_dir_path = f"../gold_data/metrics/{model_class_name}"
        metrics = {
            "MRR": round(mrr, 4),
            "Hit@10": round(hit_at_10, 4),
            "epochs": 0,
            "random_state":random_state,
            "dataset_name": str(dataset_name)
        }
        
        save_metrics(metrics, dataset_name, object_to_annotate, random_state, metric_dir_path)
        mlflow.log_metric('mrr', round(mrr, 4))
        mlflow.log_metric('hit_at_10', round(hit_at_10, 4))
        
        logger.info("Save Cross Model")
        registered_model_name = f"{dataset_name}-{object_to_annotate}-{model_class_name}"
        mlflow.sklearn.log_model(link_predictor_model, model_class_name, registered_model_name=registered_model_name)
    

if __name__ == '__main__':

 
    yaml_file_path = "./input_yaml_config/model_configs.yaml"
    yaml_file = load_yaml(yaml_file_path)
    yaml_args = yaml_file.get("xgboost_classifier_model_syn_sem_graph_similarity_learning_args", {})

    parser = argparse.ArgumentParser("XGBoost Classifier Model Parser")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--object_to_annotate', type=str)
    parser.add_argument('--random_state_index', type=int)
    parser.add_argument('--batch_size', type=int, default=yaml_args.get('batch_size'))
    parser.add_argument('--num_workers', type=int, default=yaml_args.get('num_workers'))
    parser.add_argument('--nb_epochs', type=int, default=yaml_args.get('max_epochs'))
    parser.add_argument('--num_classes', type=int, default=yaml_args.get('num_classes'))
    parser.add_argument('--top_k', type=int, default=yaml_args.get('top_k'))
    
    args, _ = parser.parse_known_args()
    main(args)
            