import torch
import logging
import argparse
import os
import pandas as pd
import string


def encode_syntactic_textual_data(sequences_src, sequences_tgt, vectorizer):
    sequences = list(sequences_src) + list(sequences_tgt)
    tfidf_matrix =  vectorizer.fit_transform(sequences).toarray()
    tfidf_src = tfidf_matrix[:sequences_src.shape[0]]
    tfidf_tgt = tfidf_matrix[sequences_src.shape[0]:]
    return tfidf_src, tfidf_tgt


def load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state):

    files_dir_path = f"{embeddings_dir_path}/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"

    list_name_files = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']
    
    for name_file in list_name_files:
        file_path = f"{files_dir_path}/{name_file}"
        if os.path.isfile(file_path):
            yield torch.load(file_path)
        else:
            print(f'File not found: {file_path}')


def load_processed_data(data_dir_path, dataset_name, object_to_predict, random_state):

    files_dir_path = f"{data_dir_path}/dataset_name={dataset_name}/object_to_predict={object_to_predict}/random_state={random_state}"

    list_file_names = [
        'train_col_alignments.parquet',
        'test_col_alignments.parquet',
        'train_ds_alignments.parquet',
        'test_ds_alignments.parquet',
        'ds_to_col.parquet',
        'ds_to_be.parquet',
        'be_to_be.parquet',
        'business_glossary_items.parquet'
    ]

    for file_name in list_file_names:

        file_path = f"{files_dir_path}/{file_name}"
        if os.path.isfile(file_path):
            yield pd.read_parquet(file_path)
        else:
            yield pd.DataFrame()

def test_mrr_hits_syntactic_model(
    test_pos_edge_index,
    obj_syn_embeddings,
    be_syn_embeddings,
    k=10,
    device=None
    ):
    
    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    obj_syn_embeddings = obj_syn_embeddings.to(device)
    be_syn_embeddings = be_syn_embeddings.to(device)
    
    with torch.no_grad():
        
        mrrs = []
        hits_at_k = []

        for i in range(test_pos_edge_index.size(1)):  # Iterate over each test edge (positive)

            # Get the source and target nodes of the positive test edge
            src, tgt = test_pos_edge_index[0, i], test_pos_edge_index[1, i]

            # Compute the score for all possible links from src to all target entities
            src_to_entities_edge_index = torch.stack([src.repeat(all_entity_ids.size(0)), all_entity_ids], dim=0).to(device)
            
            src_w_embed = obj_syn_embeddings[src_to_entities_edge_index[0]]
            trg_w_embed = be_syn_embeddings[src_to_entities_edge_index[1]]
            
            cosine_similarity = torch.nn.functional.cosine_similarity(src_w_embed, trg_w_embed)
            
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



def preprocess(sentence, stemmer, stop_words):
    # Convert to lowercase
    sentence = str(sentence).lower()

    # Remove punctuation
    sentence = sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Remove stopwords and apply stemming
    if len(words) != 1:
        processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    else:
        processed_words = words

    # Join the processed words back into a single string
    return ' '.join(processed_words)


def store_embeddings(col_embeddings, ds_embeddings, be_embeddings, dataset_name, model_type, random_state):
    
    list_objects = [col_embeddings, ds_embeddings, be_embeddings]
    list_object_names = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']

    obj_path = f"/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"
    
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
                     
    for obj, obj_name, in zip(list_objects, list_object_names):
        obj = torch.from_numpy(obj)
        torch.save(obj, f"{obj_path}/{obj_name}")


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
        "top_k":args.top_k,
        "nb_epochs":0
        }
    logger.info(args)

    random_state = [42, 84, 13][random_state_index]
        
    logger.info('Load processed data')
    data_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/raw_to_dataframes"
    data_out = list(load_processed_data(data_dir_path, dataset_name, object_to_predict, random_state))
    train_col_alignments = data_out[0]
    test_col_alignments = data_out[1]
    train_ds_alignments = data_out[2]
    test_ds_alignments = data_out[3]
    ds_to_col = data_out[4]
    ds_to_be = data_out[5]
    be_to_be = data_out[6]
    business_glossary = data_out[7]

    logger.info("Load Syntactic Embeddings")
    model_type = "syntactic-based"
    embeddings_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/embeddings"
    embeddings_out = list(load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state))
    col_syn_embeddings = embeddings_out[0]
    ds_syn_embeddings = embeddings_out[1]
    be_syn_embeddings = embeddings_out[2] 

    logger.info("Set device to 'cpu' or 'cuda'")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    logger.info("Test Syntactic Model. Testing: MRR, Hit@10")

    if object_to_predict == 'column':

        test_pos_col_edge_index =  torch.from_numpy(test_col_alignments[test_col_alignments['is_matching']==1][['col_id', 'be_id']].values).T

        mrr, hit_at_10 = test_mrr_hits_syntactic_model(
            test_pos_col_edge_index, 
            col_syn_embeddings,
            be_syn_embeddings,
            k=parameters["top_k"],
            device=device
            )
    else:
        test_pos_ds_edge_index =  torch.from_numpy(test_ds_alignments[test_ds_alignments['is_matching']==1][['ds_id', 'be_id']].values).T

        mrr, hit_at_10 = test_mrr_hits_syntactic_model(
            test_pos_ds_edge_index,
            ds_syn_embeddings,
            be_syn_embeddings,
            k=parameters["top_k"],
            device=device
            )
        

    logger.info(f"MRR: {mrr:.4f}, Hit@10: {hit_at_10:.4f}")

    logger.info("Save metrics")
    
    metric_dir_path = "/home/aknouchea/link-prediction-experiments/hybrid-link-prediction/gold_data/metrics/syntactic-model"
    metrics = {
        "MRR": round(mrr, 4),
        "Hit@10": round(hit_at_10, 4),
        "epochs": parameters["nb_epochs"],
        "random_state":random_state,
        "dataset_name": str(dataset_name)
    }
    save_metrics(metrics, dataset_name, object_to_predict, random_state, metric_dir_path)

    

    