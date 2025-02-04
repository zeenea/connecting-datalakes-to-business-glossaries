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

def test_mrr_hits_k_hybrid_sim_based(
    hybrid_model,
    test_pos_edge_index,
    obj_sem_embeddings,
    obj_syn_embeddings,
    be_sem_embeddings,
    be_syn_embeddings,
    k=10,
    device=None
    ):
    
    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data
    #print(all_entity_ids)
    obj_sem_embeddings = obj_sem_embeddings.to(device)
    obj_syn_embeddings = obj_syn_embeddings.to(device)
    be_sem_embeddings = be_sem_embeddings.to(device)
    be_syn_embeddings = be_syn_embeddings.to(device)
    
    hybrid_model.eval()

    with torch.no_grad():
        
        mrrs = []
        hits_at_k = []

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)

            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]
            #print(src)
            #print(tgt)

            # Compute the score for all possible links from src to all target entities
            src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            #print(f"src_to_entity: {src_to_entities_edge_index}")
            
            obj_sem_embed_i = obj_sem_embeddings[src_to_entities_edge_index[0]]
            obj_syn_embed_i = obj_syn_embeddings[src_to_entities_edge_index[0]]    
            be_sem_embed_i = be_sem_embeddings[src_to_entities_edge_index[1]]
            be_syn_embed_i = be_syn_embeddings[src_to_entities_edge_index[1]]
            
            logits = hybrid_model.forward(obj_sem_embed_i, obj_syn_embed_i, be_sem_embed_i, be_syn_embed_i)#[:, 1]
            #print(f"logits shape: {logits.shape}")
            #print(f"logits: {logits}")
            
            # Rank the scores (higher is better), and get the rank of the true edge
            sorted_scores, sorted_indices = torch.sort(torch.flatten(logits), descending=True)
            #print(f"sorted_scores: {sorted_scores}")
            #print(f"sorted_indices: {sorted_indices}")
            
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


def save_model(model, models_dir_path, trained_on_dataset, object_to_predict, trained_for_epochs, model_name, random_state):
    model_dir = f"{models_dir_path}/model_name={model_name}/trained_on={trained_on_dataset}/object_to_predict={object_to_predict}/random_state={random_state}/epochs={trained_for_epochs}"
    
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

    logger.info("Create Dataset and DataLoader")

    train_dataset = TextualLinkEmbeddingsDataset(train_textual_links['text'], train_textual_links['is_matching'], tokenizer, max_length=parameters['max_length'])
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=parameters['batch_size'], num_workers=parameters['num_workers'])

    logger.info("Tensorboard SummaryWriter Instatiation")
    writer_log_dir = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/trainings/{model_class_name}/dataset_name={dataset_name}/random_state={random_state}/epochs={parameters['max_epochs']}"

    if not os.path.exists(writer_log_dir):
        os.makedirs(writer_log_dir)
    
    writer = SummaryWriter(writer_log_dir)
    
    logger.info("Training. Train Loss, F1-score, Recall, Precision")
    link_predictor = train_model_on_binary_cross_entropy_loss(link_predictor, optimizer, parameters['num_classes'], parameters['max_epochs'], train_loader, writer, logger, device)
    
    # test mrr hit@10

    # save metrics


    


