import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import SAGEConv, HeteroConv, GAT
from sklearn.metrics import roc_auc_score
import logging
import os
import mlflow

os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['TORCH_USE_CUDA_DSA']='1'

torch.manual_seed(0)


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
        print("Add Implements : column -> business_entity")    
        
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
        print("Add Contains: dataset -> column")
        
        ds_to_col_pos_edge_index = torch.from_numpy(ds_to_col.values).T
        dataset['dataset', 'contains', 'column'].edge_index = ds_to_col_pos_edge_index
        dataset['column', 'rev_contains', 'dataset'].edge_index = torch.flipud(ds_to_col_pos_edge_index)
        
    if add_ds_to_be:
        print("Add Implements : dataset -> business_entity")    
        
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
        print("Add Composes: business_entity -> business_entity")
        
        be_to_be_pos_edge_index = torch.from_numpy(be_to_be.values).T.type(torch.int64)
        dataset['business_entity', 'composes', 'business_entity'].edge_index = be_to_be_pos_edge_index
        dataset['business_entity', 'rev_composes', 'business_entity'].edge_index = torch.flipud(be_to_be_pos_edge_index)

    return dataset, train_pos_col_edge_index, train_neg_col_edge_index, test_pos_col_edge_index, test_neg_col_edge_index, train_pos_ds_edge_index, train_neg_ds_edge_index, test_pos_ds_edge_index, test_neg_ds_edge_index, ds_to_col_pos_edge_index, be_to_be_pos_edge_index


from torch_geometric.nn import SAGEConv, HeteroConv, GAT

class HeteroGraphSage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):

        super(HeteroGraphSage, self).__init__()
        
        object_name = 'column'
        dataset_name = 'dataset'
        be_name = 'business_entity'
        
        implements_rel = 'implements'
        rev_implements_rel = 'rev_implements'
        contains_rel = 'contains'
        rev_contains_rel = 'rev_contains'
        composes_rel ='composes'
        rev_composes_rel = 'rev_composes'
        
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                (object_name, implements_rel, be_name): SAGEConv(in_channels, out_channels),
                (be_name, rev_implements_rel, object_name): SAGEConv(in_channels, out_channels),
                (dataset_name, contains_rel, object_name): SAGEConv(in_channels, out_channels),
                (object_name, rev_contains_rel, dataset_name): SAGEConv(in_channels, out_channels),
                (dataset_name, implements_rel, be_name): SAGEConv(in_channels, out_channels),
                (be_name, rev_implements_rel, dataset_name): SAGEConv(in_channels, out_channels),
                (be_name, composes_rel, be_name): SAGEConv(in_channels, out_channels),
                (be_name, rev_composes_rel, be_name): SAGEConv(in_channels, out_channels)
            })

            self.convs.append(conv)


    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            z_dict = conv(x_dict, edge_index_dict)
            z_dict = {key: F.relu(x) for key, x in z_dict.items()}

        return z_dict

    def decode(self, embeddings1, embeddings2):
        return (embeddings1 * embeddings2).sum(dim=-1)

    def forward(self, z_dict, edge_index_dict, source, relation, target):
        embeddings1 = z_dict[source][edge_index_dict[source, relation, target][0]] 
        embeddings2 = z_dict[target][edge_index_dict[source, relation, target][1]]
        return self.decode(embeddings1, embeddings2)


def train(hetero_model, optimizer, object_to_annotate, dataset, train_pos_col_edge_index, train_neg_col_edge_index, train_pos_ds_edge_index, train_neg_ds_edge_index, ds_to_col_pos_edge_index=None, be_to_be_pos_edge_index=None, device=None):
    
    x_dict = {}
    x_dict['column'] = dataset['column'].x.to(device)
    x_dict['business_entity'] = dataset['business_entity'].x.to(device)
    x_dict['dataset'] = dataset['dataset'].x.to(device)



    pos_edge_index_dict = {}
    
    pos_edge_index_dict['dataset', 'contains', 'column'] = ds_to_col_pos_edge_index.to(device)
    pos_edge_index_dict['column', 'rev_contains', 'dataset'] = torch.flipud(ds_to_col_pos_edge_index).to(device)
    
    neg_edge_index_dict = {}

    neg_edge_index_dict['dataset', 'contains', 'column'] = ds_to_col_pos_edge_index.to(device)
    neg_edge_index_dict['column', 'rev_contains', 'dataset'] = torch.flipud(ds_to_col_pos_edge_index).to(device)
        
    pos_edge_index_dict['business_entity', 'composes', 'business_entity'] = be_to_be_pos_edge_index.to(device)
    pos_edge_index_dict['business_entity', 'rev_composes', 'business_entity'] = torch.flipud(be_to_be_pos_edge_index).to(device)

    neg_edge_index_dict['business_entity', 'composes', 'business_entity'] = be_to_be_pos_edge_index.to(device)
    neg_edge_index_dict['business_entity', 'rev_composes', 'business_entity'] = torch.flipud(be_to_be_pos_edge_index).to(device)

    
    if object_to_annotate == 'column':
        neg_edge_index_dict['column', 'implements', 'business_entity'] = train_neg_col_edge_index.to(device)
        neg_edge_index_dict['business_entity', 'rev_implements', 'column'] = torch.flipud(train_neg_col_edge_index).to(device)

            
        pos_edge_index_dict['column', 'implements', 'business_entity'] = train_pos_col_edge_index.to(device)
        pos_edge_index_dict['business_entity','rev_implements','column'] = torch.flipud(train_pos_col_edge_index).to(device)

    
    if object_to_annotate == 'dataset':
        
        pos_edge_index_dict['dataset', 'implements', 'business_entity'] = train_pos_ds_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(train_pos_ds_edge_index).to(device)

        neg_edge_index_dict['dataset', 'implements', 'business_entity'] = train_neg_ds_edge_index.to(device)
        neg_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(train_neg_ds_edge_index).to(device)

        

    hetero_model.train()
    optimizer.zero_grad()

    
    # positive edge scores
    pos_z_dict = hetero_model.encode(x_dict, pos_edge_index_dict)

    if object_to_annotate == 'column':
        pos_scores = hetero_model.forward(pos_z_dict, pos_edge_index_dict, source='column', relation='implements', target='business_entity')
        pos_labels = torch.ones(pos_scores.size(0)).to(device)

    if object_to_annotate == 'dataset':
        pos_scores = hetero_model.forward(pos_z_dict, pos_edge_index_dict, source='dataset', relation='implements', target='business_entity')
        pos_labels = torch.ones(pos_scores.size(0)).to(device)

    #scores = torch.concat([pos_scores_col_to_be, pos_scores_ds_to_be])
    #labels = torch.concat([pos_labels_col_to_be, pos_scores_ds_to_be])

    # negative edge scores
    neg_z_dict = hetero_model.encode(x_dict, neg_edge_index_dict)

    if object_to_annotate == 'column':
        neg_scores = hetero_model.forward(neg_z_dict, neg_edge_index_dict, source='column', relation='implements', target='business_entity')
        neg_labels = torch.zeros(neg_scores.size(0)).to(device)

    if object_to_annotate == 'dataset':
        neg_scores = hetero_model.forward(neg_z_dict, neg_edge_index_dict, source='dataset', relation='implements', target='business_entity')
        neg_labels = torch.zeros(neg_scores.size(0)).to(device)

    #scores = torch.concat([scores, neg_scores_col_to_be, neg_scores_ds_to_be])
    #labels = torch.concat([labels, neg_labels_col_to_be, neg_scores_ds_to_be])

    
    scores = torch.concat([pos_scores, neg_scores]).to(device)
    labels = torch.concat([pos_labels, neg_labels]).to(device)
    
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    loss.backward()
    optimizer.step()

    return loss, pos_edge_index_dict
        

def test(hetero_model, object_to_annotate, dataset, test_pos_col_edge_index, test_neg_col_edge_index, test_pos_ds_edge_index, test_neg_ds_edge_index, ds_to_col_pos_edge_index=None, be_to_be_pos_edge_index=None, device=None):
    
    hetero_model.eval()

    with torch.no_grad():
        x_dict = {}
        x_dict['column'] = dataset.x_dict['column'].to(device)
        x_dict['business_entity'] = dataset.x_dict['business_entity'].to(device)
        x_dict['dataset'] = dataset['dataset'].x.to(device)

        pos_edge_index_dict = {}
        neg_edge_index_dict = {}


        pos_edge_index_dict['dataset', 'contains', 'column'] = ds_to_col_pos_edge_index.to(device)
        pos_edge_index_dict['column', 'rev_contains', 'dataset'] = torch.flipud(ds_to_col_pos_edge_index).to(device)

        neg_edge_index_dict['dataset', 'contains', 'column'] = ds_to_col_pos_edge_index.to(device)
        neg_edge_index_dict['column', 'rev_contains', 'dataset'] = torch.flipud(ds_to_col_pos_edge_index).to(device)

        pos_edge_index_dict['business_entity', 'composes', 'business_entity'] = be_to_be_pos_edge_index.to(device)
        pos_edge_index_dict['business_entity', 'rev_composes', 'business_entity'] = torch.flipud(be_to_be_pos_edge_index).to(device)

        neg_edge_index_dict['business_entity', 'composes', 'business_entity'] = be_to_be_pos_edge_index.to(device)
        neg_edge_index_dict['business_entity', 'rev_composes', 'business_entity'] = torch.flipud(be_to_be_pos_edge_index).to(device)


        if object_to_annotate == 'column':

            pos_edge_index_dict['column','implements','business_entity'] = test_pos_col_edge_index.to(device)
            pos_edge_index_dict['business_entity','rev_implements','column'] = torch.flipud(test_pos_col_edge_index).to(device)
    
            neg_edge_index_dict['column', 'implements', 'business_entity'] = test_neg_col_edge_index.to(device)
            neg_edge_index_dict['business_entity', 'rev_implements', 'column'] = torch.flipud(test_neg_col_edge_index).to(device)


        if object_to_annotate=='dataset':

            pos_edge_index_dict['dataset', 'implements', 'business_entity'] = test_pos_ds_edge_index.to(device)
            pos_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(test_pos_ds_edge_index).to(device)

            neg_edge_index_dict['dataset', 'implements', 'business_entity'] = test_neg_ds_edge_index.to(device)
            neg_edge_index_dict['business_entity', 'rev_implements', 'dataset'] = torch.flipud(test_neg_ds_edge_index).to(device)


        
            
            
        # positive edge scores
        pos_z_dict = hetero_model.encode(x_dict, pos_edge_index_dict)

        if object_to_annotate == 'column':
            pos_scores = hetero_model.forward(pos_z_dict, pos_edge_index_dict, source='column', relation='implements', target='business_entity')
            pos_labels = torch.ones(pos_scores.size(0)).to(device)
        

        if object_to_annotate == 'dataset':
            pos_scores = hetero_model.forward(pos_z_dict, pos_edge_index_dict, source='dataset', relation='implements', target='business_entity')
            pos_labels = torch.ones(pos_scores.size(0)).to(device)
        
        
        # negative edge scores
        neg_z_dict = hetero_model.encode(x_dict, neg_edge_index_dict)
        
        if object_to_annotate == 'column':
            neg_scores = hetero_model.forward(neg_z_dict, neg_edge_index_dict, source='column', relation='implements', target='business_entity')
            neg_labels = torch.zeros(neg_scores.size(0)).to(device)
        

        if object_to_annotate == 'dataset':
            neg_scores = hetero_model.forward(neg_z_dict, neg_edge_index_dict, source='dataset', relation='implements', target='business_entity')
            neg_labels = torch.zeros(neg_scores.size(0)).to(device)
        
    
        # Calcul de la perte (Binary Cross-Entropy pour une tâche de classification binaire)
        scores = torch.cat([pos_scores, neg_scores]).to(device)
        labels = torch.cat([pos_labels, neg_labels]).to(device)
        
        # Compute AUC for evaluation
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        auc = roc_auc_score(labels.cpu(), scores.cpu())

    return loss, auc


def test_mrr_hits_k(col_embeddings, ds_embeddings, be_embeddings, source_object, dataset, hetero_model, train_pos_edge_index, test_pos_edge_index, k=10, device=None):
    hetero_model.eval()
    
    all_entity_ids = test_pos_edge_index[1,:].unique() # get the right test data, and load the right data

    with torch.no_grad():
        # Encode node embeddings using the trained GraphSAGE model
        
        x_dict = {}
        
        #for key, value in dataset.x_dict.items():
        #    x_dict[key] = value.to(device)

        x_dict['column'] = col_embeddings.to(device)
        x_dict['dataset'] = ds_embeddings.to(device)
        x_dict['business_entity'] = be_embeddings.to(device)

        edge_index_dict = {}

        #for key, value in dataset.edge_index_dict.items():
        #    print(key)
        #    print(value)
        #    edge_index_dict[key] = value.long().to(device)
            
        z = hetero_model.encode(x_dict, train_pos_edge_index)

        # Prepare lists to accumulate MRR and Hit@K results
        mrrs = []
        hits_at_k = []

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
            _, sorted_indices = torch.sort(edge_scores, descending=True)

            # Get the rank of the true edge (index 0 corresponds to the true edge score)
            #true_edge_rank = (sorted_indices == 0).nonzero(as_tuple=False).item() + 1  # rank starts from 1
            true_edge_rank = (sorted_indices == target_true_index).nonzero(as_tuple=True)[0].item()
            
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

def assertion_verification_on_edge_indexes(
object_to_annotate,
dataset,
train_pos_col_edge_index,
train_neg_col_edge_index,
test_pos_col_edge_index,
test_neg_col_edge_index,
train_pos_ds_edge_index,
train_neg_ds_edge_index,
test_pos_ds_edge_index,
test_neg_ds_edge_index,
ds_to_col_pos_edge_index,
be_to_be_pos_edge_index
):
    assert max(train_pos_col_edge_index[0]) <= dataset['column'].x.shape[0]
    assert max(train_pos_col_edge_index[1]) <= dataset['business_entity'].x.shape[0]
    assert max(train_neg_col_edge_index[0]) <= dataset['column'].x.shape[0]
    assert max(train_neg_col_edge_index[1]) <= dataset['business_entity'].x.shape[0]
    
    if object_to_annotate == 'column':
        assert max(test_pos_col_edge_index[0]) <= dataset['column'].x.shape[0]
        assert max(test_pos_col_edge_index[1]) <= dataset['business_entity'].x.shape[0]
        assert max(test_neg_col_edge_index[0]) <= dataset['column'].x.shape[0]
        assert max(test_neg_col_edge_index[1]) <= dataset['business_entity'].x.shape[0]
    
    assert max(train_pos_ds_edge_index[0]) <= dataset['dataset'].x.shape[0]
    assert max(train_pos_ds_edge_index[1]) <= dataset['business_entity'].x.shape[0]
    
    if object_to_annotate == 'dataset':
        assert max(train_neg_ds_edge_index[0]) <= dataset['dataset'].x.shape[0]
        assert max(train_neg_ds_edge_index[1]) <= dataset['business_entity'].x.shape[0]
    
        assert max(test_pos_ds_edge_index[0]) <= dataset['dataset'].x.shape[0]
        assert max(test_pos_ds_edge_index[1]) <= dataset['business_entity'].x.shape[0]
        assert max(test_neg_ds_edge_index[0]) <= dataset['dataset'].x.shape[0]
        assert max(test_neg_ds_edge_index[1]) <= dataset['business_entity'].x.shape[0]

    assert max(ds_to_col_pos_edge_index[0]) <= dataset['dataset'].x.shape[0]
    assert max(ds_to_col_pos_edge_index[1]) <= dataset['column'].x.shape[0]

    assert max(be_to_be_pos_edge_index[0]) <= dataset['business_entity'].x.shape[0]
    assert max(be_to_be_pos_edge_index[1]) <= dataset['business_entity'].x.shape[0]

            
def save_metrics(metrics:dict, dataset_name, object_to_annotate, random_state, metric_dir):

            if not os.path.exists(metric_dir):
                os.makedirs(metric_dir)
                
            metric_file = open(f"{metric_dir}/link_prediction_{dataset_name}_{object_to_annotate}_{random_state}.txt", "w")
            metric_file.write(str(metrics))
            metric_file.close()


def save_model(model, models_dir_path, trained_on_dataset, trained_for_epochs, model_name, random_state):
        model_dir = f"{models_dir_path}/trained_on={trained_on_dataset}/random_state={random_state}/epochs={trained_for_epochs}"
        
        if os.path.exists(model_dir):
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
        else:
            os.makedirs(model_dir)
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
    

def save_torch_tensor(tensor, tensor_dir_path, tensor_name):
    
        torch.save(tensor, f"{tensor_dir_path}/{tensor_name}")


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
    max_epochs = args.max_epochs
    graph_dim_embeddings = args.graph_dim_embeddings
    lr = args.learning_rate

    logger.info(args)

    random_state = [42, 84, 13][random_state_index]

    logger.info('Load embeddings')
    model_type = "semantic-based"
    embeddings_dir_path = "../gold_data/embeddings"
    embeddings_out = list(load_embeddings(embeddings_dir_path, dataset_name, model_type, random_state))
    col_embeddings = embeddings_out[0]
    ds_embeddings = embeddings_out[1]
    be_embeddings = embeddings_out[2] 
        
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

    logger.info('Create hetero graph dataset')

    add_col_to_be = True
    add_ds_to_col = True
    add_ds_to_be = True
    add_be_to_be = True
    
    edge_indexes_out = create_hetero_graph_dataset(
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
    )

    hetero_dataset = edge_indexes_out[0]
    train_pos_col_edge_index = edge_indexes_out[1]
    train_neg_col_edge_index = edge_indexes_out[2]
    test_pos_col_edge_index = edge_indexes_out[3]
    test_neg_col_edge_index = edge_indexes_out[4]
    train_pos_ds_edge_index = edge_indexes_out[5]
    train_neg_ds_edge_index = edge_indexes_out[6]
    test_pos_ds_edge_index = edge_indexes_out[7]
    test_neg_ds_edge_index = edge_indexes_out[8]
    ds_to_col_pos_edge_index = edge_indexes_out[9]
    be_to_be_pos_edge_index = edge_indexes_out[10]

    logger.info('Assertion verification on edge indexes')
    assertion_verification_on_edge_indexes(
        object_to_annotate,
        hetero_dataset,
        train_pos_col_edge_index,
        train_neg_col_edge_index,
        test_pos_col_edge_index,
        test_neg_col_edge_index,
        train_pos_ds_edge_index,
        train_neg_ds_edge_index,
        test_pos_ds_edge_index,
        test_neg_ds_edge_index,
        ds_to_col_pos_edge_index,
        be_to_be_pos_edge_index
    )
    
    logger.info('Define device')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device= {device}")
    
    logger.info('Creates HeteroGraphSage Model')
    assert col_embeddings.shape[1] == ds_embeddings.shape[1]
    assert col_embeddings.shape[1] == be_embeddings.shape[1]
    
    in_channels = col_embeddings.shape[1]
    out_channels = graph_dim_embeddings
    
    hetero_model = HeteroGraphSage(in_channels=in_channels, out_channels=out_channels).to(device)

    model_class_name = hetero_model.__class__.__name__
    
    optimizer = torch.optim.Adam(hetero_model.parameters(), lr=lr)

    options = dict()
    
    if add_ds_to_col:
        options['ds_to_col_pos_edge_index'] = ds_to_col_pos_edge_index
    else:
        options['ds_to_col_pos_edge_index'] = None
    
    if add_be_to_be:
        options['be_to_be_pos_edge_index'] = be_to_be_pos_edge_index
    else:
        options['be_to_be_pos_edge_index'] = None

    options['device'] = device

    logger.info("MLFlow managing")
    mlflow.set_experiment('hetero_graph_model')
    
    with mlflow.start_run():
        
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag('object_to_annotate', object_to_annotate)
        mlflow.log_param('max_epochs', max_epochs)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('dataset_split_random_state', random_state)
        mlflow.log_param('loss_function', 'binary_cross_entropy_with_logits')
        mlflow.log_param('optimizer', 'AdamW')
        mlflow.log_param('link_predictor_model', model_class_name)
    
        logger.info("Training - Train Loss - Test Loss - Test AUC")
        # early stopping params
        best_loss = float('inf')
        patience = 5
        min_delta = 1e-5
        patience_counter = 0
        
        for epoch in range(0, max_epochs):

            train_loss, train_pos_edge_index = train(hetero_model, optimizer, object_to_annotate, hetero_dataset, train_pos_col_edge_index, train_neg_col_edge_index, train_pos_ds_edge_index, train_neg_ds_edge_index, **options)
            test_loss, test_auc  = test(hetero_model, object_to_annotate, hetero_dataset, test_pos_col_edge_index, test_neg_col_edge_index, test_pos_ds_edge_index, test_neg_ds_edge_index, **options)
            logger.info(f'Epoch: {epoch}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
            
            train_metrics = {
                'train_loss': round(train_loss.item(), 4),
                'test_loss': round(test_loss.item(), 4),
                'test_auc': round(test_auc.item(), 4)
            }
            mlflow.log_metrics(train_metrics, epoch)

            if train_loss.item() < best_loss - min_delta:
                best_loss = train_loss.item()
                patience_counter = 0
                best_model_state = hetero_model.state_dict()
            else:
                patience_counter +=1
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        hetero_model.load_state_dict(best_model_state)

        logger.info("Testing - Hit@10 - MRR")
        if object_to_annotate == 'column':
            mrr, hit_at_10 = test_mrr_hits_k(col_embeddings, ds_embeddings, be_embeddings, object_to_annotate, hetero_dataset, hetero_model, train_pos_edge_index, test_pos_col_edge_index, k=10, device=device)
            
        elif object_to_annotate == 'dataset':
            mrr, hit_at_10 = test_mrr_hits_k(col_embeddings, ds_embeddings, be_embeddings, object_to_annotate, hetero_dataset, hetero_model, train_pos_edge_index, test_pos_ds_edge_index, k=10, device=device) 
    
        
        logger.info(f"MRR: {mrr:.4f}, Hit@10: {hit_at_10:.4f}")
    
        logger.info("Save metrics")
        
        metric_dir_path = "../gold_data/metrics/hetero-graph-model"
        
        metrics = {
            "MRR": round(mrr, 4),
            "Hit@10": round(hit_at_10, 4),
            "epochs": max_epochs,
            "random_state":random_state,
            "dataset_name": str(dataset_name)
        }
        
        save_metrics(metrics, dataset_name, object_to_annotate, random_state, metric_dir_path)

        mlflow.log_metric('mrr', round(mrr, 4))
        mlflow.log_metric('hit_at_10', round(hit_at_10, 4))
    
        logger.info("Save HeteroGraph Model")
        models_dir_path = "../gold_data/models"
        model_name = "heteroGraphSage"
        
        save_model(hetero_model, models_dir_path, dataset_name, max_epochs, model_name, random_state)

        registered_model_name = f"{dataset_name}-{object_to_annotate}-{model_class_name}"
        mlflow.pytorch.log_model(hetero_model, model_class_name, registered_model_name=registered_model_name)

        logger.info("Save Graph Embeddings")
    
        x_dict = {}
    
        x_dict['column'] = col_embeddings.to(device)
        x_dict['dataset'] = ds_embeddings.to(device)
        x_dict['business_entity'] = be_embeddings.to(device)
           
        graph_embeddings = hetero_model.encode(x_dict, train_pos_edge_index)
    
        col_graph_embeddings = graph_embeddings['column']
        ds_graph_embeddings = graph_embeddings['dataset']
        be_graph_embeddings = graph_embeddings['business_entity']
    
        graph_embeddings_dir_path = f"../gold_data/embeddings/dataset_name={dataset_name}/model_type=graph-based/random_state={random_state}"
    
        if not os.path.exists(graph_embeddings_dir_path):
                os.makedirs(graph_embeddings_dir_path)
            
        save_torch_tensor(col_graph_embeddings, graph_embeddings_dir_path, 'col_embeddings.pt')
        save_torch_tensor(ds_graph_embeddings, graph_embeddings_dir_path, 'ds_embeddings.pt')
        save_torch_tensor(be_graph_embeddings, graph_embeddings_dir_path, 'be_embeddings.pt')
        
        logger.info("Save edge indexes")
        edge_indexes_dir_path = f"../gold_data/edge_indexes/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"
    
        if not os.path.exists(edge_indexes_dir_path):
                os.makedirs(edge_indexes_dir_path)
    
        save_torch_tensor(train_pos_col_edge_index, edge_indexes_dir_path, 'train_pos_col_edge_index.pt')
        save_torch_tensor(train_neg_col_edge_index, edge_indexes_dir_path, 'train_neg_col_edge_index.pt')
        save_torch_tensor(test_pos_col_edge_index, edge_indexes_dir_path, 'test_pos_col_edge_index.pt')
        save_torch_tensor(test_neg_col_edge_index, edge_indexes_dir_path, 'test_neg_col_edge_index.pt')
        save_torch_tensor(train_pos_ds_edge_index, edge_indexes_dir_path, 'train_pos_ds_edge_index.pt')
        save_torch_tensor(train_neg_ds_edge_index, edge_indexes_dir_path, 'train_neg_ds_edge_index.pt')
        save_torch_tensor(test_pos_ds_edge_index, edge_indexes_dir_path, 'test_pos_ds_edge_index.pt')
        save_torch_tensor(test_neg_ds_edge_index, edge_indexes_dir_path, 'test_neg_ds_edge_index.pt')
        save_torch_tensor(ds_to_col_pos_edge_index, edge_indexes_dir_path, 'ds_to_col_pos_edge_index.pt')
        save_torch_tensor(be_to_be_pos_edge_index, edge_indexes_dir_path, 'be_to_be_pos_edge_index.pt')
            
    
    
