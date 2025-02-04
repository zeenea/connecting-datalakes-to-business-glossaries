import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer 
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import logging
import os

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_semantic_textual_link_embeddings(dataset_name, object_to_predict, random_state):

    embeddings_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings/dataset_name={dataset_name}/model_type=semantic-based/random_state={random_state}/"

    embedding_file_names = [f"train_{object_to_predict}_be_textual_link_embeddings.pt", f"test_{object_to_predict}_be_textual_link_embeddings.pt"]

    for file_name in embedding_file_names:
        file_path = f"{embeddings_path}/{file_name}"
        if os.path.exists(file_path):
            yield torch.load(file_path)


def load_textual_links(dataset_name, object_to_predict, random_state):
    obj_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/raw_to_dataframes/dataset_name={dataset_name}/object_to_predict={object_to_predict}/random_state={random_state}"

    file_names = ['train_textual_links.parquet', 'test_textual_links.parquet']
    
    for file_name in file_names:
        file_path = f"{obj_path}/{file_name}"
        if os.path.exists(file_path):
            yield pd.read_parquet(file_path)

                        
class TextualLinkEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.max_length = max_length

    def __getitem__(self, item):
        tokens = self.tokenizer(
            self.sentences[item],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = self.labels[item]

        return tokens["input_ids"], tokens["attention_mask"], torch.tensor(label)

    def __len__(self):
        return len(self.sentences)

def collate_fn(batch):
    input_ids = torch.stack([x[0] for x in batch])
    attention_mask = torch.stack([x[1] for x in batch])
    labels = torch.stack([x[2] for x in batch])

    return input_ids, attention_mask, labels


class TextualLinkDatasetForInference(torch.utils.data.Dataset):

    def __init__(self, source_to_target, source_id, source_name, target_id, name_id_target, tokenizer, max_length):

        self.source_to_target = source_to_target
        self.name_id_target = name_id_target 
        self.source_id = source_id
        self.source_name = source_name
        self.target_id = target_id

        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, item):
        
        #print(f"item: {item}")
        #print(f"source_id: {self.source_id}")
        #print(f"source_name: {self.source_name}")
        #print(f"len source_to_target: {len(self.source_to_target)}")
        
        src_id = self.source_to_target.loc[item, self.source_id]
        src_name = self.source_to_target.loc[item, self.source_name]
        true_tgt_id = self.source_to_target.loc[item, self.target_id]
        #print(f"src_id: {src_id}")
        #print(f"src_name: {src_name}")
        
        df_tmp = self.name_id_target
        df_tmp[self.source_name] = src_name
        df_tmp[self.source_id] = src_id
        df_tmp['true_tgt_id'] = true_tgt_id
        df_tmp['text'] = df_tmp.apply(lambda x: f"[CLS]{str(x[self.source_name])}[SEP]{str(x['be_name'])}[SEP]", axis=1)

        tokens = self.tokenizer(
            df_tmp['text'].values.tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return torch.from_numpy(df_tmp[self.source_id].values), torch.from_numpy(df_tmp['be_id'].values), torch.from_numpy(df_tmp['true_tgt_id'].values), tokens['input_ids'], tokens['attention_mask']

    def __len__(self):
        return len(self.source_to_target)

def collate_test_dataset_fn(batch):

    source_ids = torch.stack([x[0] for x in batch])
    target_ids = torch.stack([x[1] for x in batch])
    true_tgt_id = torch.stack([x[2] for x in batch])
    input_ids = torch.stack([x[3] for x in batch])
    attention_mask = torch.stack([x[4] for x in batch])  

    return source_ids, target_ids, true_tgt_id, input_ids, attention_mask
    

class BinaryClassifierModel(torch.nn.Module):

    def __init__(self, model_embeddings, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model_emebddings = model_embeddings
        self.linear = torch.nn.Linear(model_embeddings.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
       
    def forward(self, input_ids, attention_mask):
        outputs = self.model_emebddings(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        token_embeddings = outputs['hidden_states'][-1]

        input_mask_expanded = (
            attention_mask.squeeze(1)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
        )
        vector_embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        logits = self.sigmoid(self.linear(vector_embeddings))
        return logits


def train_model_on_binary_cross_entropy_loss(link_predictor, optimizer, num_classes, max_epochs, edge_loader, writer, logger, device):
    link_predictor.train()
    
    precision_func = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_func = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    f1score_func = MulticlassF1Score( num_classes=num_classes, average=None).to(device)

    loss_criterion = torch.nn.BCELoss()

    for epoch in range(max_epochs):

        loss_epoch = 0
        loss_batch  = []
        f1score = torch.tensor([]).to(device)
        precision = torch.tensor([]).to(device)
        recall = torch.tensor([]).to(device)

    
        for batch in edge_loader:
    
            input_ids = batch[0].squeeze(1).to(device)
            attention_mask = batch[1].squeeze(1).to(device)
            labels = batch[2].to(device)
            
            #labels = torch.flatten(labels).to(device)
            labels = labels.reshape(-1, 1).float()
    
            optimizer.zero_grad()

            logits = link_predictor.forward(input_ids, attention_mask)

            #loss = torch.nn.functional.binary_cross_entropy(logits, labels)
            loss = loss_criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())
            # todo: implement early stoping
            
            
            # compute precision recall f1 score on training
            with torch.no_grad():
                predictions = torch.where(logits >= 0.5, 1, 0)
                f1score = torch.concat([f1score, f1score_func(predictions, labels.long()).reshape(1, -1)], axis=0)
                precision = torch.concat([precision, precision_func(predictions, labels.long()).reshape(1, -1)], axis=0)
                recall = torch.concat([recall, recall_func(predictions, labels.long()).reshape(1, -1)], axis=0)

                
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

    return link_predictor




def test_mrr_hits_k(
    link_predictor,
    #test_pos_alignments,
    #obj_sem_embeddings,
    #obj_syn_embeddings,
    #be_sem_embeddings,
    #be_syn_embeddings,
    test_loader,
    k=10,
    device=None
    ):
    
    #all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data
    #print(all_entity_ids)
    #obj_sem_embeddings = obj_sem_embeddings.to(device)
    #obj_syn_embeddings = obj_syn_embeddings.to(device)
    #be_sem_embeddings = be_sem_embeddings.to(device)
    #be_syn_embeddings = be_syn_embeddings.to(device)
    
    link_predictor.eval()

    with torch.no_grad():
        
        mrrs = []
        hits_at_k = []

        #for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)
        for batch in test_loader:

            # Get the source and target nodes of the positive test edge
            #src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]
            #print(src)
            #print(tgt)

            # Compute the score for all possible links from src to all target entities
            #src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            #print(f"src_to_entity: {src_to_entities_edge_index}")
            
            #obj_sem_embed_i = obj_sem_embeddings[src_to_entities_edge_index[0]]
            #obj_syn_embed_i = obj_syn_embeddings[src_to_entities_edge_index[0]]    
            #be_sem_embed_i = be_sem_embeddings[src_to_entities_edge_index[1]]
            #be_syn_embed_i = be_syn_embeddings[src_to_entities_edge_index[1]]
            src_ids = batch[0].to(device)
            tgt_ids = batch[1].to(device)
            true_tgt_id = batch[2].to(device)
            input_ids = batch[3].to(device)
            input_ids = input_ids.view(-1, input_ids.shape[2])
            attention_mask = batch[4].to(device)
            attention_mask = attention_mask.view(-1, attention_mask.shape[2])

            print(true_tgt_id)
            print(src_ids)
            print(tgt_ids)
            
            
            logits = link_predictor.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.view(src_ids.shape[0],-1)
            print(f"logits shape: {logits.shape}")
            print(f"logits: {logits}")
            
            # Rank the scores (higher is better), and get the rank of the true edge
            #sorted_scores, sorted_indices = torch.sort(torch.flatten(logits), descending=True)
            sorted_scores, sorted_indices = torch.sort(logits, descending=True, axis=1)
            print(f"sorted_scores: {sorted_scores}")
            print(f"sorted_indices: {sorted_indices}")
            
            # get sorted entity ids by score
            for sub_tgt_ids, sub_sorted_indices in zip(tgt_ids, sorted_indices, list_tgt):
                
                sorted_entity_ids = sub_tgt_ids[sub_sorted_indice]
                print(f"sorted_entity_ids: {sorted_entity_ids}")
    
                # get rank of true target entity 
                true_edge_rank = (sorted_entity_ids == tgt).nonzero(as_tuple=True)[0].item()
                
                # MRR Calculation: Reciprocal of the true edge's rank
                mrrs.append(1.0 / (true_edge_rank+1))
    
                # Hit@K Calculation: Check if the true edge appears in the top K scores
                if true_edge_rank <= k:
                    hits_at_k.append(1.0)  # 1 means hit
                else:
                    hits_at_k.append(0.0)  # 0 means miss

            
            break

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


def save_model(model, models_dir_path, trained_on_dataset, object_to_predict, trained_for_epochs, model_name, random_state):
    model_dir = f"{models_dir_path}/model_name={model_name}/trained_on={trained_on_dataset}/object_to_predict={object_to_predict}/random_state={random_state}/epochs={trained_for_epochs}"
    
    if os.path.exists(model_dir):
        torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
    else:
        os.makedirs(model_dir)
        torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")


def load_processed_data(data_dir_path, dataset_name, object_to_predict, random_state):

    files_dir_path = f"{data_dir_path}/dataset_name={dataset_name}/object_to_predict={object_to_predict}/random_state={random_state}"

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
    object_to_predict = args.object_to_predict
    random_state_index = args.random_state_index

    parameters = {
        "batch_size":args.batch_size,
        "num_workers":args.num_workers,
        "max_epochs":args.max_epochs,
        "num_classes":args.num_classes,
        "learning_rate":args.learning_rate,
        "top_k":args.top_k,
        "max_length": 64
    }

    logger.info(args)
    
    random_state = [42, 84, 13][random_state_index]

    logger.info("Set device to 'cpu' or 'cuda'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    logger.info("Load Semantic Textual Link Embeddings")
    textual_links = list(load_textual_links(dataset_name, object_to_predict, random_state))

    train_textual_links = textual_links[0]
    test_textual_links = textual_links[1]
    
    logger.info("Load Textual Encoder model and tokenizer")

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_embeddings = AutoModel.from_pretrained(model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Initialize Binary Classifier Model and Optimizer")

    link_predictor = BinaryClassifierModel(
        model_embeddings = model_embeddings,
        num_classes = parameters['num_classes']
    )

    model_class_name = link_predictor.__class__.__name__

    link_predictor.to(device)

    optimizer = torch.optim.AdamW(link_predictor.parameters(), lr=parameters['learning_rate'])

    logger.info("Create Train Dataset and DataLoader")

    train_dataset = TextualLinkEmbeddingsDataset(train_textual_links['text'], train_textual_links['is_matching'], tokenizer, max_length=parameters['max_length'])
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])

    logger.info("Tensorboard SummaryWriter Instatiation")
    writer_log_dir = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/trainings/{model_class_name}/dataset_name={dataset_name}/random_state={random_state}/epochs={parameters['max_epochs']}"

    if not os.path.exists(writer_log_dir):
        os.makedirs(writer_log_dir)
    
    writer = SummaryWriter(writer_log_dir)
    
    logger.info("Training. Train Loss, F1-score, Recall, Precision")
    #link_predictor = train_model_on_binary_cross_entropy_loss(link_predictor, optimizer, parameters['num_classes'], parameters['max_epochs'], train_loader, writer, logger, device)    
    
    logger.info("Create Test Dataset and DataLoader")

    name_id_target = test_textual_links[['be_id', 'be_name']].drop_duplicates(subset=['be_id']).reset_index(drop=True)
    if object_to_predict == 'column':
        source_name = "column_name"
        source_id = "col_id"
        target_id = 'be_id'

        pos_test_textual_links = test_textual_links[test_textual_links['is_matching']==1].reset_index(drop=True)
        test_dataset = TextualLinkDatasetForInference(pos_test_textual_links, source_id, source_name, target_id, name_id_target, tokenizer, max_length=parameters['max_length'])
    else:
        source_name = "table_name"
        source_id = "ds_id"
        target_id = 'be_id'

        pos_test_textual_links = test_textual_links[test_textual_links['is_matching']==1].reset_index(drop=True)
        test_dataset = TextualLinkDatasetForInference(pos_test_textual_links, source_id, source_name, target_id, name_id_target, tokenizer, max_length=parameters['max_length'])
        
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_test_dataset_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])

    logger.info("Testing. MRR and Hit@10")

    mrr, hit_at_k = test_mrr_hits_k(link_predictor, test_loader, parameters['top_k'], device)

    logger.info(f"MRR: {mrr:.4f}, Hit@10: {hit_at_10:.4f}")

    logger.info("Save metrics")
    
    metric_dir_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/metrics/{model_class_name}"
    metrics = {
        "MRR": round(mrr, 4),
        "Hit@10": round(hit_at_10, 4),
        "epochs": parameters["nb_epochs"],
        "random_state":random_state,
        "dataset_name": str(dataset_name)
    }
    
    save_metrics(metrics, dataset_name, object_to_predict, random_state, metric_dir_path)

    logger.info("Save Binary Classifier Model")
    models_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/models"
    
    save_model(link_predictor, models_dir_path, dataset_name, object_to_predict, parameters['nb_epochs'], model_class_name, random_state)

    


