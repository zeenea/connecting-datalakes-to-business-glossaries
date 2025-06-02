import os 
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import logging
import string
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



def load_business_glossary(business_glossary_path, dataset_name=None):

    if os.path.isfile(business_glossary_path):

        business_glossary = pd.read_csv(business_glossary_path, index_col=0)
        if dataset_name =='turl-cta':
            
            bgi_tmp = pd.DataFrame()
            bgi_tmp['business_entity_code'] = business_glossary['business_entity_code'].apply(lambda x: str(x).split('.')[0])
            bgi_tmp['business_entity_name'] = bgi_tmp['business_entity_code']
            business_glossary = pd.concat([business_glossary, bgi_tmp], axis=0)
            business_glossary = business_glossary.reset_index(drop=True)

        return business_glossary

    elif os.path.isdir(business_glossary_path):

        business_glossary = pd.DataFrame()
        business_glossaries_file_names = os.listdir(business_glossary_path)
        for file_path in business_glossaries_file_names:
            if '.dvc' not in file_path:
                business_glossary = pd.concat([business_glossary, pd.read_csv(business_glossary_path+"/"+file_path, index_col=0)], axis=0).reset_index(drop=True)

        return business_glossary


def encode_semantic_textual_data(squences, model):
    return model.encode(squences)


def preprocess(sentence, stemmer, stop_words):
    # Convert to lowercase
    sentence = str(sentence).lower()

    # Remove punctuation
    sentence = sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Remove stopwords and apply stemming
    if len(words) > 1:
        processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    else:
        processed_words = words

    # Join the processed words back into a single string
    return ' '.join(processed_words)


def store_embeddings(col_embeddings, ds_embeddings, be_embeddings, dataset_name, model_type, random_state):
    
    list_objects = [col_embeddings, ds_embeddings, be_embeddings]
    list_object_names = ['col_embeddings.pt', 'ds_embeddings.pt', 'be_embeddings.pt']

    obj_path = f"../gold_data/embeddings/dataset_name={dataset_name}/model_type={model_type}/random_state={random_state}"
    
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
                     
    for obj, obj_name, in zip(list_objects, list_object_names):
        obj = torch.from_numpy(obj)
        torch.save(obj, f"{obj_path}/{obj_name}")

def store_dataframe(data, data_save_name, dataset_name, object_to_annotate, random_state):
    
    obj_path = f"../gold_data/raw_to_dataframes/dataset_name={dataset_name}/object_to_annotate={object_to_annotate}/random_state={random_state}"
    
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

                     
    data.to_parquet(f"{obj_path}/{data_save_name}")


def load_t2dv2_artifacts(neg_strategy, random_state):
    pos_obj_alignments_path =  "../gold_data/raw_data/t2dv2/alignments/column_to_business_glossary_alignments.csv"
    neg_obj_alignments_path = f"../gold_data/raw_data/t2dv2/negative-alignments/{neg_strategy}_neg_column_alignments.csv"

    pos_obj_alignments = pd.read_csv(pos_obj_alignments_path, index_col=0)
    pos_obj_alignments = pos_obj_alignments[~pos_obj_alignments['target_uri'].isnull()]
    pos_obj_alignments = pos_obj_alignments.reset_index(drop=True)
    pos_obj_alignments = pos_obj_alignments.reset_index()
    pos_obj_alignments = pos_obj_alignments.rename(columns={'index':'col_id'})
    pos_obj_alignments['is_matching'] = 1

    neg_obj_alignments = pd.read_csv(neg_obj_alignments_path, index_col=0)
    neg_obj_alignments = neg_obj_alignments.rename(columns={'neg_business_entity_code':'target_uri'})
    neg_obj_alignments = neg_obj_alignments[~neg_obj_alignments['target_uri'].isnull()]
    neg_obj_alignments = neg_obj_alignments.reset_index(drop=True)
    neg_obj_alignments = neg_obj_alignments.reset_index()
    neg_obj_alignments = neg_obj_alignments.rename(columns={'index':'col_id'})
    neg_obj_alignments['is_matching'] = 0

    all_obj_alignments = pd.concat([pos_obj_alignments, neg_obj_alignments], axis=0)
    all_obj_alignments = all_obj_alignments.sample(frac=1, random_state=random_state).reset_index(drop=True)

    assert pos_obj_alignments.shape[0] == neg_obj_alignments.shape[0]
    assert max(all_obj_alignments['col_id']) < all_obj_alignments.shape[0] 

    train_size = int(0.8 * len(pos_obj_alignments)) 

    train_pos_alignments = pos_obj_alignments[:train_size]
    train_neg_alignments = neg_obj_alignments[:train_size]

    test_pos_alignments = pos_obj_alignments[train_size:]
    test_neg_alignments = neg_obj_alignments[train_size:]

    train_alignments = pd.concat([train_pos_alignments, train_neg_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_alignments = pd.concat([test_pos_alignments, test_neg_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


    business_glossary_items_path = "../gold_data/raw_data/t2dv2/business-glossary/property_metadata.csv"
    business_glossary_items = load_business_glossary(business_glossary_items_path)
    business_glossary_items = business_glossary_items.reset_index(drop=True)
    business_glossary_items = business_glossary_items.reset_index()
    business_glossary_items = business_glossary_items.rename(columns={'index':'be_id'})
    business_glossary_items['be_id'] = business_glossary_items['be_id'].astype(int)
    assert max(business_glossary_items['be_id']) +1  == business_glossary_items.shape[0]

    # load dataset alignments
    dataset_alignments_path = "../gold_data/raw_data/t2dv2/alignments/dataset_to_business_glossary_alignments.csv"
    dataset_alignments = pd.read_csv(dataset_alignments_path)
    dataset_alignments['id_table'] = dataset_alignments['table_file'].apply(lambda x: x[:-7])
    dataset_alignments.loc[dataset_alignments[dataset_alignments['uri']=="http://www.w3.org/2002/07/owl#Thing"].index, 'uri'] = None 
    dataset_alignments = dataset_alignments.reset_index(drop=True)
    dataset_alignments = dataset_alignments.reset_index()
    dataset_alignments = dataset_alignments.rename(columns={'index':'ds_id', 'uri':'target_uri'})
    dataset_alignments['is_matching'] = 1
    assert max(dataset_alignments['ds_id']) +1 == dataset_alignments.shape[0]

    dataset_alignments = pd.merge(dataset_alignments, pos_obj_alignments[['col_id', 'id_table']], left_on='id_table', right_on='id_table', how='right')
    dataset_alignments = dataset_alignments.drop(columns=['ds_id'])
    dataset_alignments = dataset_alignments.reset_index(drop=True)
    dataset_alignments = dataset_alignments.reset_index()
    dataset_alignments = dataset_alignments.rename(columns={'index': 'ds_id'})
    
    ds_to_obj = dataset_alignments[['ds_id', 'col_id']]    

    domain_glossary_items = business_glossary_items[business_glossary_items['domain'].notnull()]
    domain_glossary_items = domain_glossary_items[['domain', 'domain_label', 'domain_comment']]
    domain_glossary_items = domain_glossary_items[domain_glossary_items['domain'].notna()]
    domain_glossary_items = domain_glossary_items.drop_duplicates(subset=['domain'])
    domain_glossary_items = domain_glossary_items.reset_index(drop=True)
    domain_glossary_items = domain_glossary_items.reset_index()
    domain_glossary_items = domain_glossary_items.rename(columns={'index':'dm_id'})
    assert max(domain_glossary_items['dm_id']) +1 == domain_glossary_items.shape[0]

    business_glossary_items = business_glossary_items[['uri', 'label', 'comment', 'domain', 'domain_label', 'domain_comment']]
    domain_glossary_items = domain_glossary_items[['domain', 'domain_label', 'domain_comment']]
    domain_glossary_items.columns = ['uri', 'label', 'comment']
    domain_glossary_items['domain'] = ""
    domain_glossary_items['domain_label'] = ""
    domain_glossary_items['domain_comment'] = ""

    business_glossary_items = pd.concat([business_glossary_items, domain_glossary_items], axis=0)
    business_glossary_items = business_glossary_items.reset_index(drop=True)
    business_glossary_items = business_glossary_items.reset_index()
    business_glossary_items = business_glossary_items.rename(columns={'index':'be_id'})

    be_source_tmp = business_glossary_items[business_glossary_items['domain']!= ""]
    be_source_tmp = be_source_tmp.reset_index(drop=True)
     
    be_target_tmp = business_glossary_items[business_glossary_items['domain'] == ""]
    be_target_tmp = be_target_tmp.reset_index(drop=True)

    be_to_be = pd.merge(be_source_tmp[['be_id', 'domain']], be_target_tmp[['be_id', 'uri']].rename(columns={'be_id':'dm_id'}), left_on='domain', right_on='uri', how='inner')

    be_to_be = be_to_be[['be_id', 'dm_id']]
    be_to_be = be_to_be[be_to_be['dm_id'].notnull()]
    be_to_be = be_to_be.reset_index(drop=True)
    be_to_be['dm_id'] = be_to_be['dm_id'].astype(int)
    
    train_alignments = pd.merge(train_alignments, business_glossary_items[['be_id', 'uri']], left_on='target_uri', right_on='uri', how='inner')
    test_alignments = pd.merge(test_alignments, business_glossary_items[['be_id', 'uri']], left_on='target_uri', right_on='uri', how='inner')

    train_ds_alignments = pd.merge(dataset_alignments, business_glossary_items, left_on='target_uri', right_on='uri', how='left')
    test_ds_alignments = pd.DataFrame()

    train_ds_alignments = train_ds_alignments.rename(columns={'table_file':'table_name'})

    business_glossary_items = business_glossary_items.rename(columns={'label':'be_name'})
    dataset_alignments = dataset_alignments.rename(columns={'table_file':'table_name'})
    
    return train_alignments, test_alignments, train_ds_alignments, test_ds_alignments, business_glossary_items, ds_to_obj, be_to_be, pos_obj_alignments, dataset_alignments


def load_zeenea_open_ds_artifacts(train_on, random_state):

    dataset_name = 'zeenea-open-ds'
        
    pos_col_alignments_path = "../gold_data/raw_data/zeenea-open-ds/alignments/column_to_business_glossary_alignments.csv"
    neg_col_alignments_path = "../gold_data/raw_data/zeenea-open-ds/negative-alignments/random_neg_column_alignments.csv"
    
    pos_col_alignments = pd.read_csv(pos_col_alignments_path, index_col=0)
    pos_col_alignments = pos_col_alignments[~pos_col_alignments['business_entity_code'].isnull()]
    pos_col_alignments = pos_col_alignments.reset_index(drop=True)
    pos_col_alignments = pos_col_alignments.reset_index()
    pos_col_alignments = pos_col_alignments.rename(columns={'index':'col_id'})
    pos_col_alignments['is_matching'] = 1
    
    neg_col_alignments = pd.read_csv(neg_col_alignments_path, index_col=0)
    neg_col_alignments = neg_col_alignments.rename(columns={'neg_business_entity_code':'business_entity_code'})
    neg_col_alignments = neg_col_alignments[~neg_col_alignments['business_entity_code'].isnull()]
    neg_col_alignments = neg_col_alignments.reset_index(drop=True)
    neg_col_alignments = neg_col_alignments.reset_index()
    neg_col_alignments = neg_col_alignments.rename(columns={'index':'col_id'})
    neg_col_alignments['is_matching'] = 0
    
    assert pos_col_alignments.shape[0] == neg_col_alignments.shape[0]

    if train_on =='column':
        train_size = int(0.8 * len(pos_col_alignments)) 
    else:
        train_size = len(pos_col_alignments)

    train_pos_col_alignments = pos_col_alignments[:train_size]
    train_neg_col_alignments = neg_col_alignments[:train_size]

    if train_on == 'column':
        test_pos_col_alignments = pos_col_alignments[train_size:]
        test_neg_col_alignments = neg_col_alignments[train_size:]

    train_col_alignments = pd.concat([train_pos_col_alignments, train_neg_col_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    if train_on == 'column':
        test_col_alignments = pd.concat([test_pos_col_alignments, test_neg_col_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


    business_glossary_items_path = "../gold_data/raw_data/zeenea-open-ds/business-glossaries"
    business_glossary_items = load_business_glossary(business_glossary_items_path)
    business_glossary_items = business_glossary_items.reset_index(drop=True)
    business_glossary_items = business_glossary_items.reset_index()
    business_glossary_items = business_glossary_items.rename(columns={'index':'be_id'})
    business_glossary_items['be_id'] = business_glossary_items['be_id'].apply(lambda x: int(x))
    
    
    train_col_alignments = pd.merge(train_col_alignments, business_glossary_items[['be_id', 'code']], left_on='business_entity_code', right_on='code', how='inner')
    assert train_col_alignments[train_col_alignments['is_matching']==0].shape[0] > 0
    assert train_col_alignments[train_col_alignments['is_matching']==1].shape[1] > 0

    if train_on == 'column':
        test_col_alignments = pd.merge(test_col_alignments, business_glossary_items[['be_id', 'code']], left_on='business_entity_code', right_on='code', how='inner')
        assert test_col_alignments[test_col_alignments['is_matching']==0].shape[0] > 0
        assert test_col_alignments[test_col_alignments['is_matching']==1].shape[0] > 0
        
    # be_to_be
    be_to_be = business_glossary_items[['be_id', 'sub_class_of']]
    be_to_be = be_to_be.rename(columns={'be_id': 'be_src'})
    be_to_be = pd.merge(be_to_be, business_glossary_items[['be_id','code']], left_on='sub_class_of', right_on='code', how='left')[['be_src', 'be_id']]
    be_to_be = be_to_be[be_to_be['be_id'].notnull()]
    be_to_be = be_to_be.reset_index(drop=True)
    
    # load dataset alignments
    pos_ds_alignments_path = "../gold_data/raw_data/zeenea-open-ds/alignments/dataset_to_business_glossary_alignments.csv"
    pos_ds_alignments = pd.read_csv(pos_ds_alignments_path, index_col=0)
    pos_ds_alignments = pos_ds_alignments.reset_index(drop=True)
    pos_ds_alignments = pos_ds_alignments.reset_index()
    pos_ds_alignments = pos_ds_alignments.rename(columns={'index':'ds_id'})
    pos_ds_alignments['is_matching'] = 1
    
    assert max(pos_ds_alignments['ds_id']) < (pos_ds_alignments.shape[0])
    
    # ds_to_col
    ds_to_col = pos_col_alignments[['col_id', 'dataset_name']]
    ds_to_col = pd.merge(ds_to_col, pos_ds_alignments[['ds_id', 'dataset_code']], left_on='dataset_name', right_on='dataset_code', how='inner')
    ds_to_col = ds_to_col[['ds_id', 'col_id']]
    ds_to_col = ds_to_col.reset_index(drop=True)

    business_glossary_items = business_glossary_items.rename(columns={'name':'be_name'})
    pos_ds_alignments = pos_ds_alignments.rename(columns={'dataset_code':'table_name'})
    
    # split test and train for dataset alignments
    neg_ds_alignments_path = "../gold_data/raw_data/zeenea-open-ds/negative-alignments/random_neg_dataset_alignments.csv"
    neg_ds_alignments = pd.read_csv(neg_ds_alignments_path, index_col=0)
    neg_ds_alignments = neg_ds_alignments.rename(columns={'neg_business_entity_code':'business_entity_code'})
    neg_ds_alignments = neg_ds_alignments[~neg_ds_alignments['business_entity_code'].isnull()]
    neg_ds_alignments = neg_ds_alignments.reset_index(drop=True)
    neg_ds_alignments = neg_ds_alignments.reset_index()
    neg_ds_alignments = neg_ds_alignments.rename(columns={'index':'ds_id'})
    neg_ds_alignments['is_matching'] = 0

    neg_ds_alignments = neg_ds_alignments.rename(columns={'dataset_code':'table_name'})
    
    if train_on == 'dataset':
        train_size = int(0.8 * len(pos_ds_alignments)) 
    else:
        train_size = len(pos_ds_alignments)

    train_pos_ds_alignments = pos_ds_alignments[:train_size]
    train_neg_ds_alignments = neg_ds_alignments[:train_size]

    if train_on == 'dataset':
        test_pos_ds_alignments = pos_ds_alignments[train_size:]
        test_neg_ds_alignments = neg_ds_alignments[train_size:]

    train_ds_alignments = pd.concat([train_pos_ds_alignments, train_neg_ds_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if train_on == 'dataset':
        test_ds_alignments = pd.concat([test_pos_ds_alignments, test_neg_ds_alignments], axis=0).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_ds_alignments = pd.merge(train_ds_alignments, business_glossary_items[['be_id', 'code']], left_on='business_entity_code', right_on='code', how='inner')
    
    if train_on == 'dataset':
        test_ds_alignments = pd.merge(test_ds_alignments, business_glossary_items[['be_id', 'code']], left_on='business_entity_code', right_on='code', how='inner')


    if train_on == 'column':
        test_ds_alignments = None
    elif train_on == 'dataset':
        test_col_alignments = None
    
    return train_col_alignments, test_col_alignments, train_ds_alignments, test_ds_alignments, business_glossary_items, ds_to_col, be_to_be, pos_col_alignments, pos_ds_alignments


def load_turl_cta_artifacts(train_on, random_state):

    dataset_name = 'turl-cta'
    
    # train 
    train_pos_col_alignments_path = "../gold_data/raw_data/turl-cta/alignments/column-alignments/train_column_alignments.csv"
    train_pos_col_alignments = pd.read_csv(train_pos_col_alignments_path, index_col=0)
    train_pos_col_alignments = train_pos_col_alignments.rename(columns={'column_label':'target_uri'})
    train_pos_col_alignments = train_pos_col_alignments.reset_index(drop=True)
    train_pos_col_alignments = train_pos_col_alignments[['table_id', 'target_uri', 'column_name']]
    train_pos_col_alignments['is_matching'] = 1

    train_neg_col_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/columns/train_random_neg_alignments.csv"
    train_neg_col_alignments = pd.read_csv(train_neg_col_alignments_path, index_col=0)
    train_neg_col_alignments = train_neg_col_alignments.rename(columns={'neg_business_entity_code':'target_uri'})
    train_neg_col_alignments = train_neg_col_alignments.reset_index(drop=True)
    train_neg_col_alignments = train_neg_col_alignments[['table_id', 'target_uri', 'column_name']]
    train_neg_col_alignments['is_matching'] = 0
    
    assert train_pos_col_alignments.shape[0] == train_neg_col_alignments.shape[0]

    # dev
    dev_pos_col_alignments_path = "../gold_data/raw_data/turl-cta/alignments/column-alignments/dev_column_alignments.csv"
    dev_pos_col_alignments = pd.read_csv(dev_pos_col_alignments_path, index_col=0)
    dev_pos_col_alignments = dev_pos_col_alignments.rename(columns={'column_label':'target_uri'})
    dev_pos_col_alignments = dev_pos_col_alignments.reset_index(drop=True)
    dev_pos_col_alignments = dev_pos_col_alignments[['table_id', 'target_uri', 'column_name']]
    dev_pos_col_alignments['is_matching'] = 1

    dev_neg_col_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/columns/dev_random_neg_alignments.csv"
    dev_neg_col_alignments = pd.read_csv(dev_neg_col_alignments_path, index_col=0)
    dev_neg_col_alignments = dev_neg_col_alignments.rename(columns={'neg_business_entity_code':'target_uri'})
    dev_neg_col_alignments = dev_neg_col_alignments.reset_index(drop=True)
    dev_neg_col_alignments = dev_neg_col_alignments[['table_id', 'target_uri', 'column_name']]
    dev_neg_col_alignments['is_matching'] = 0
    

    assert dev_pos_col_alignments.shape[0] == dev_neg_col_alignments.shape[0]

    # test
    test_pos_col_alignments_path = "../gold_data/raw_data/turl-cta/alignments/column-alignments/test_column_alignments.csv"
    test_pos_col_alignments = pd.read_csv(test_pos_col_alignments_path, index_col=0)
    test_pos_col_alignments = test_pos_col_alignments.rename(columns={'column_label':'target_uri'})
    test_pos_col_alignments = test_pos_col_alignments.reset_index(drop=True)
    test_pos_col_alignments = test_pos_col_alignments[['table_id', 'target_uri', 'column_name']]
    test_pos_col_alignments['is_matching'] = 1
    
    test_neg_col_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/columns/test_random_neg_alignments.csv"
    test_neg_col_alignments = pd.read_csv(test_neg_col_alignments_path, index_col=0)
    test_neg_col_alignments = test_neg_col_alignments.rename(columns={'neg_business_entity_code':'target_uri'})
    test_neg_col_alignments = test_neg_col_alignments.reset_index(drop=True)
    test_neg_col_alignments = test_neg_col_alignments[['table_id', 'target_uri', 'column_name']]
    test_neg_col_alignments['is_matching'] = 0

    assert test_pos_col_alignments.shape[0] == test_neg_col_alignments.shape[0]


    # pos and neg alignments

    pos_col_alignments = pd.concat([train_pos_col_alignments, dev_pos_col_alignments, test_pos_col_alignments], axis=0)
    pos_col_alignments = pos_col_alignments[~pos_col_alignments['column_name'].isna()]
    pos_col_alignments = pos_col_alignments.reset_index(drop=True)
    pos_col_alignments = pos_col_alignments.reset_index()
    pos_col_alignments = pos_col_alignments.rename(columns={'index':'col_id'})
    pos_col_alignments = pos_col_alignments.sample(frac=1, random_state=random_state)

    neg_col_alignments = pd.concat([train_neg_col_alignments, dev_neg_col_alignments, test_neg_col_alignments], axis=0)
    neg_col_alignments = neg_col_alignments[~neg_col_alignments['column_name'].isna()]
    neg_col_alignments = neg_col_alignments.reset_index(drop=True)
    neg_col_alignments = neg_col_alignments.reset_index()
    neg_col_alignments = neg_col_alignments.rename(columns={'index':'col_id'})
    neg_col_alignments = neg_col_alignments.sample(frac=1, random_state=random_state)

    assert pos_col_alignments.shape[0] == neg_col_alignments.shape[0]
    
    # train and test sets
    if train_on == 'column':
        train_size = int(pos_col_alignments.shape[0] * 0.8)
    else:
        train_size = pos_col_alignments.shape[0]
        
    train_col_alignments = pd.concat([pos_col_alignments[:train_size], neg_col_alignments[:train_size]], axis=0).sample(frac=1)

    if train_on == 'column':
        test_col_alignments = pd.concat([pos_col_alignments[train_size:], neg_col_alignments[train_size:]], axis=0).sample(frac=1)
    
    # load business glossary
    business_glossary_path = "../gold_data/raw_data/turl-cta/business-glossary/glossary_data.csv"
    business_glossary_items = load_business_glossary(business_glossary_path)
    business_glossary_items = business_glossary_items.reset_index(drop=True)
    business_glossary_items['sub_class_of'] = business_glossary_items['business_entity_code'].apply(lambda x: str(x).split('.')[0])
    parent_entities = business_glossary_items['sub_class_of'].unique()
    business_glossary_items = pd.concat([pd.DataFrame({'business_entity_code':parent_entities, 'business_entity_name': parent_entities, 'sub_class_of':[None for i in range(len(parent_entities))]}), business_glossary_items], axis=0)
    business_glossary_items = business_glossary_items.reset_index(drop=True)
    business_glossary_items = business_glossary_items.reset_index()
    business_glossary_items = business_glossary_items.rename(columns={'index':'be_id'})
    business_glossary_items['be_id'] = business_glossary_items['be_id'].apply(lambda x: int(x))
    
    assert max(business_glossary_items['be_id']) < (business_glossary_items.shape[0])
    
    be_code_bg = "business_entity_code"

    train_col_alignments = pd.merge(train_col_alignments, business_glossary_items[['be_id', be_code_bg]], left_on='target_uri', right_on=be_code_bg, how='inner')
    train_col_alignments = train_col_alignments[['col_id', 'column_name', 'be_id', 'is_matching']]

    if train_on == 'column':
        test_col_alignments = pd.merge(test_col_alignments, business_glossary_items[['be_id', be_code_bg]], left_on='target_uri', right_on=be_code_bg, how='inner')
        test_col_alignments = test_col_alignments[['col_id', 'column_name', 'be_id', 'is_matching']]
    
    # be_to_be
    be_to_be = business_glossary_items[['be_id', 'sub_class_of']]
    be_to_be = be_to_be.rename(columns={'be_id':'src_be_id'})
    be_to_be = pd.merge(be_to_be, business_glossary_items[['be_id', 'business_entity_code']], left_on='sub_class_of', right_on='business_entity_code', how='left')
    be_to_be = be_to_be[['src_be_id', 'be_id']]
    be_to_be = be_to_be.rename(columns={'be_id':'trgt_be_id'})
    be_to_be = be_to_be[be_to_be['trgt_be_id'].notnull()]
    be_to_be = be_to_be.reset_index(drop=True)

    # load dataset alignments

    # train
    train_pos_ds_alignments_path = f"../gold_data/raw_data/data/turl-cta/alignments/dataset-alignments/train_dataset_alignments.csv"
    train_pos_ds_alignments = pd.read_csv(train_pos_ds_alignments_path)
    train_pos_ds_alignments = train_pos_ds_alignments.drop_duplicates(subset=['table_id'])
    train_pos_ds_alignments = train_pos_ds_alignments.reset_index(drop=True)
    train_pos_ds_alignments = train_pos_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    train_pos_ds_alignments['is_matching'] = 1

    train_neg_ds_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/datasets/train_random_neg_alignments.csv"
    train_neg_ds_alignments = pd.read_csv(train_neg_ds_alignments_path, index_col=0)
    train_neg_ds_alignments = train_neg_ds_alignments.rename(columns={'neg_business_entity_code':'tag_entity'})
    train_neg_ds_alignments = train_neg_ds_alignments.reset_index(drop=True)
    train_neg_ds_alignments = pd.merge(train_neg_ds_alignments, train_pos_ds_alignments[['table_id', 'table_title']], left_on='table_id', right_on='table_id', how='inner')
    train_neg_ds_alignments = train_neg_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    train_neg_ds_alignments['is_matching'] = 0
    
    # dev 

    dev_pos_ds_alignments_path = f"../gold_data/raw_data/turl-cta/alignments/dataset-alignments/dev_dataset_alignments.csv"
    dev_pos_ds_alignments = pd.read_csv(train_pos_ds_alignments_path)
    dev_pos_ds_alignments = dev_pos_ds_alignments.drop_duplicates(subset=['table_id'])
    dev_pos_ds_alignments = dev_pos_ds_alignments.reset_index(drop=True)
    dev_pos_ds_alignments = dev_pos_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    dev_pos_ds_alignments['is_matching'] = 1

    dev_neg_ds_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/datasets/dev_random_neg_alignments.csv"
    dev_neg_ds_alignments = pd.read_csv(dev_neg_ds_alignments_path, index_col=0)
    dev_neg_ds_alignments = dev_neg_ds_alignments.rename(columns={'neg_business_entity_code':'tag_entity'})
    dev_neg_ds_alignments = dev_neg_ds_alignments.reset_index(drop=True)
    dev_neg_ds_alignments = pd.merge(dev_neg_ds_alignments, dev_pos_ds_alignments[['table_id', 'table_title']], left_on='table_id', right_on='table_id', how='inner')
    dev_neg_ds_alignments = dev_neg_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    dev_neg_ds_alignments['is_matching'] = 0

    
    # test
    test_pos_ds_alignments_path = f"../gold_data/raw_data/turl-cta/alignments/dataset-alignments/test_dataset_alignments.csv"
    test_pos_ds_alignments = pd.read_csv(test_pos_ds_alignments_path)
    test_pos_ds_alignments = test_pos_ds_alignments.drop_duplicates(subset=['table_id'])
    test_pos_ds_alignments = test_pos_ds_alignments.reset_index(drop=True)
    test_pos_ds_alignments = test_pos_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    test_pos_ds_alignments['is_matching'] = 1 

    test_neg_ds_alignments_path = "../gold_data/raw_data/turl-cta/negative-alignments/datasets/test_random_neg_alignments.csv"
    test_neg_ds_alignments = pd.read_csv(test_neg_ds_alignments_path, index_col=0)
    test_neg_ds_alignments = test_neg_ds_alignments.rename(columns={'neg_business_entity_code':'tag_entity'})
    test_neg_ds_alignments = test_neg_ds_alignments.reset_index(drop=True)
    test_neg_ds_alignments = pd.merge(test_neg_ds_alignments, test_pos_ds_alignments[['table_id', 'table_title']], left_on='table_id', right_on='table_id', how='inner')
    test_neg_ds_alignments = test_neg_ds_alignments[['table_id', 'tag_entity', 'table_title']]
    test_neg_ds_alignments['is_matching'] = 0

    
    pos_ds_alignments = pd.concat([train_pos_ds_alignments, dev_pos_ds_alignments, test_pos_ds_alignments], axis=0)
    pos_ds_alignments = pos_ds_alignments[~pos_ds_alignments['table_title'].isna()]
    pos_ds_alignments = pos_ds_alignments.reset_index(drop=True)
    pos_ds_alignments = pos_ds_alignments.reset_index()
    pos_ds_alignments = pos_ds_alignments.rename(columns={'index':'ds_id'})
    pos_ds_alignments = pd.merge(pos_ds_alignments[['ds_id', 'table_id', 'tag_entity', 'table_title', 'is_matching']],  business_glossary_items[['be_id', 'business_entity_code']], left_on='tag_entity', right_on='business_entity_code', how='inner')
    pos_ds_alignments = pos_ds_alignments.sample(frac=1, random_state=random_state)

    neg_ds_alignments = pd.concat([train_neg_ds_alignments, dev_neg_ds_alignments, test_neg_ds_alignments], axis=0)
    neg_ds_alignments = neg_ds_alignments[~neg_ds_alignments['table_title'].isna()]
    neg_ds_alignments = neg_ds_alignments.reset_index(drop=True)
    neg_ds_alignments = neg_ds_alignments.reset_index()
    neg_ds_alignments = neg_ds_alignments.rename(columns={'index':'ds_id'})
    neg_ds_alignments = pd.merge(neg_ds_alignments[['ds_id', 'table_id', 'tag_entity', 'table_title', 'is_matching']],  business_glossary_items[['be_id', 'business_entity_code']], left_on='tag_entity', right_on='business_entity_code', how='inner')
    neg_ds_alignments = neg_ds_alignments.sample(frac=1, random_state=random_state)

    # train and test sets
    if train_on == 'dataset':
        train_size = int(pos_ds_alignments.shape[0] * 0.8)
    else:
        train_size = pos_ds_alignments.shape[0]
    
    train_ds_alignments = pd.concat([pos_ds_alignments[:train_size], neg_ds_alignments[:train_size]], axis=0).sample(frac=1)

    if train_on == 'dataset':
        test_ds_alignments = pd.concat([pos_ds_alignments[train_size:], neg_ds_alignments[train_size:]], axis=0).sample(frac=1)

    
    # ds_to_col
    ds_to_col = pd.merge(pos_ds_alignments, pos_col_alignments, left_on='table_id', right_on='table_id', how='inner')
    ds_to_col = ds_to_col[['ds_id', 'col_id']]
    ds_to_col = ds_to_col.reset_index(drop=True)

    business_glossary_items = business_glossary_items.rename(columns={'business_entity_name':'be_name'})
    pos_ds_alignments = pos_ds_alignments.rename(columns={'table_title':'table_name'})
    neg_ds_alignments = neg_ds_alignments.rename(columns={'table_title':'table_name'})
    train_ds_alignments = train_ds_alignments.rename(columns={'table_title':'table_name'})

    
    if train_on == 'column':
        test_ds_alignments = None

    if train_on == 'dataset':
        test_col_alignments = None
        test_ds_alignments = test_ds_alignments.rename(columns={'table_title':'table_name'})
    
    return train_col_alignments, test_col_alignments, train_ds_alignments, test_ds_alignments, business_glossary_items, ds_to_col, be_to_be, pos_col_alignments, pos_ds_alignments


def save_model(model, models_dir_path, trained_on_dataset, trained_for_epochs, model_name, random_state):
        model_dir = f"{models_dir_path}/trained_on={trained_on_dataset}/random_state={random_state}/epochs={trained_for_epochs}"
        
        if os.path.exists(model_dir):
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
        else:
            os.makedirs(model_dir)
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
            

def get_text_batches(df, object_name, batch_size=10000):
    if df.shape[0] > batch_size:
        nb_chunks = df.shape[0] // batch_size
    else:
        nb_chunks = 1
        
    for chunk_idx in np.array_split(df.index, nb_chunks):
        chunk_df =  df.loc[chunk_idx]
        yield chunk_df[object_name].astype('str')
        

def update_vocabulary(df, object_name, batch_size, vocabulary):
        for batch in get_text_batches(df, object_name, batch_size):
            for text in batch:
                tokens = text.lower().split()
                vocabulary.update(tokens)
        return vocabulary


def get_embeddings(df, object_name, batch_size, vectorizer):
        for batch in get_text_batches(df, object_name, batch_size):
            yield  vectorizer.fit_transform(batch).toarray() 
            

def generate_tfidf_embeddings(col_df_file, ds_df_file, be_df_file, batch_size, logger):

    vocabulary = Counter({})

    col_name = 'column_name'
    ds_name = 'table_name'
    be_name = 'be_name'

    logger.info("--Update vocabulary")
    vocabulary = update_vocabulary(col_df_file, col_name, batch_size, vocabulary)
    vocabulary = update_vocabulary(ds_df_file, ds_name, batch_size, vocabulary)
    vocabulary = update_vocabulary(be_df_file, be_name, batch_size, vocabulary)

    logger.info("Reduce Vocabulary Size")
    max_df = 100000
    max_features = 1000

    filtered_vocab = dict(filter(lambda item: item[1] <= max_df, dict(vocabulary).items()))
    
    top_tokens_vocab = dict(sorted(filtered_vocab.items(), key=lambda item: item[1], reverse=True)[:max_features])

    logger.info("--Create Vectorizer")
    vectorizer = TfidfVectorizer(vocabulary=top_tokens_vocab.keys(), stop_words='english')#), max_features=500)#, max_df=0.8)

    logger.info("--Generate TFIDF Embeddings")
    logger.info("----Column Embeddings")
    col_embeddings = np.concatenate(list(get_embeddings(col_df_file, col_name, batch_size, vectorizer)), axis=0)
    logger.info("----Table Embeddings")
    ds_embeddings = np.concatenate(list(get_embeddings(ds_df_file, ds_name, batch_size, vectorizer)), axis=0)
    logger.info("----BE Embeddings")
    be_embeddings = np.concatenate(list(get_embeddings(be_df_file, be_name, batch_size, vectorizer)), axis=0)

    return col_embeddings, ds_embeddings, be_embeddings

def generate_semantic_embeddings(df, column_to_encode, model, stemmer, stop_words):

    return encode_semantic_textual_data(df[column_to_encode].apply(lambda x: preprocess(x, stemmer=stemmer, stop_words=stop_words)), model)


def generate_textual_link(dataframe, source_id, source_name, target_id, target_name):
    dataframe['text'] = dataframe[[source_name, target_name]].apply(lambda x: f"[CLS]{str(x[source_name])}[SEP]{str(x[target_name])}[SEP]", axis=1)
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.reset_index()
    return dataframe[['index', source_id, source_name, target_id, target_name, 'text', 'is_matching']]


def main(args):

    nltk.download('punkt')

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Load Arguments")
    
    dataset_name = args.dataset_name
    object_to_annotate = args.object_to_annotate
    random_state_index = args.random_state_index
    neg_strategy = args.neg_strategy
    generate_syntactic_embeddings_bool = args.generate_syntactic_embeddings
    generate_semantic_embeddings_bool = args.generate_semantic_embeddings
    generate_semantic_textual_links_bool = args.generate_semantic_textual_links

    logger.info(args)
    
    list_random_states = [42, 84, 13]
    random_state = list_random_states[random_state_index]

    logger.info("Set device to 'cpu' or 'cuda' ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    logger.info("Load Data Artefacts")
    if dataset_name == 't2dv2':
        results = load_t2dv2_artifacts(neg_strategy, random_state)
        train_col_alignments = results[0]
        test_col_alignments = results[1]
        train_ds_alignments = results[2]
        test_ds_alignments = results[3]
        business_glossary_items = results[4]
        ds_to_col = results[5]
        be_to_be = results[6]
        pos_col_alignments = results[7]
        pos_ds_alignments = results[8]

    if dataset_name == "zeenea-open-ds":
    
        results = load_zeenea_open_ds_artifacts(object_to_annotate, random_state)
        train_col_alignments = results[0]
        test_col_alignments = results[1]
        train_ds_alignments = results[2]
        test_ds_alignments = results[3]
        business_glossary_items = results[4]
        ds_to_col = results[5]
        be_to_be = results[6]
        pos_col_alignments = results[7]
        pos_ds_alignments = results[8]


    if dataset_name == "turl-cta":
        
        results = load_turl_cta_artifacts(object_to_annotate, random_state)
        train_col_alignments = results[0]
        test_col_alignments = results[1]
        train_ds_alignments = results[2]
        test_ds_alignments = results[3]
        business_glossary_items = results[4]
        ds_to_col = results[5]
        be_to_be = results[6]
        pos_col_alignments = results[7]
        pos_ds_alignments = results[8]

    
    logger.info("Save Data Artefacts")
    store_dataframe(train_col_alignments, "train_col_alignments.parquet", dataset_name, object_to_annotate, random_state)
    if object_to_annotate == 'column':
        store_dataframe(test_col_alignments, "test_col_alignments.parquet", dataset_name, object_to_annotate, random_state)
    
    store_dataframe(train_ds_alignments, "train_ds_alignments.parquet", dataset_name, object_to_annotate, random_state)
    if object_to_annotate == 'dataset':
        store_dataframe(train_ds_alignments, "test_ds_alignments.parquet", dataset_name, object_to_annotate, random_state)
    
    store_dataframe(business_glossary_items, "business_glossary_items.parquet", dataset_name, object_to_annotate, random_state)
    store_dataframe(ds_to_col, "ds_to_col.parquet", dataset_name, object_to_annotate, random_state)
    store_dataframe(be_to_be, "be_to_be.parquet", dataset_name, object_to_annotate, random_state)


    if generate_semantic_embeddings_bool or generate_semantic_textual_links_bool:
        
        logger.info("Set Stopwords")
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        logger.info("Load model")
        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name).to(device)

    if generate_semantic_embeddings_bool: 
        logger.info("Generate Semantic Embeddings")
        
        col_sem_embeddings = generate_semantic_embeddings(pos_col_alignments, "column_name", model, stemmer, stop_words) 
        ds_sem_embeddings = generate_semantic_embeddings(pos_ds_alignments, "table_name", model, stemmer, stop_words) 
        be_sem_embeddings = generate_semantic_embeddings(business_glossary_items, 'be_name', model, stemmer, stop_words) 

        logger.info("Save Semantic Embeddings")
        store_embeddings(col_sem_embeddings, ds_sem_embeddings, be_sem_embeddings, dataset_name, 'semantic-based', random_state)
    
        logger.info(f"col_sem_embeddings: {col_sem_embeddings.shape}")
        logger.info(f"ds_sem_embeddings: {ds_sem_embeddings.shape}")
        logger.info(f"be_sem_embeddings: {be_sem_embeddings.shape}")

        logger.info("Save semantic Model")
        models_dir_path = "../gold_data/models"
        save_model(model, models_dir_path, dataset_name, 0, model_name, random_state)

    
    if generate_syntactic_embeddings_bool:
        logger.info("Generate Syntactic Embeddings")
        
        syntactic_embeddings = generate_tfidf_embeddings(
            pos_col_alignments,
            pos_ds_alignments,
            business_glossary_items,
            batch_size=1000,
            logger=logger
        )
        
        col_syn_embeddings = syntactic_embeddings[0]
        ds_syn_embeddings = syntactic_embeddings[1]
        be_syn_embeddings = syntactic_embeddings[2]

        logger.info(f"col_syn_embeddings: {col_syn_embeddings.shape}")
        logger.info(f"ds_syn_embeddings: {ds_syn_embeddings.shape}")
        logger.info(f"be_syn_embeddings: {be_syn_embeddings.shape}")
    
        logger.info("Save Syntactic Embeddings")
        store_embeddings(col_syn_embeddings, ds_syn_embeddings, be_syn_embeddings, dataset_name, 'syntactic-based', random_state)

    if generate_semantic_textual_links_bool:
        logger.info("Generate Textual Links <[CLS]source_name[SEP]target_name[SEP]>")

        
        if object_to_annotate == 'column':
            column_name = "column_name"
            column_id = "col_id"
            be_name = "be_name"
            be_id = "be_id"
            
            train_col_alignments = pd.merge(train_col_alignments, business_glossary_items[[be_id, be_name]], on=be_id, how='inner')
            test_col_alignments = pd.merge(test_col_alignments, business_glossary_items[[be_id, be_name]], on=be_id, how='inner')
            print(train_col_alignments.columns)
            print(test_col_alignments.columns)
            
            train_textual_links = generate_textual_link(train_col_alignments, column_id, column_name, be_id, be_name) 
            test_textual_links = generate_textual_link(test_col_alignments, column_id, column_name, be_id, be_name)

        if object_to_annotate == 'dataset':
            table_name = "table_name"
            table_id = "ds_id"
            be_name = "be_name"
            be_id = "be_id"
            

            train_ds_alignments = pd.merge(train_ds_alignments, business_glossary_items[[be_id, be_name]], on=be_id, how='inner')
            test_ds_alignments = pd.merge(test_ds_alignments, business_glossary_items[[be_id, be_name]], on=be_id, how='inner')
            
            train_textual_links = generate_textual_link(train_ds_alignments, table_id, table_name, be_id, be_name) 
            test_textual_links = generate_textual_link(test_ds_alignments, table_id, table_name, be_id, be_name)

        logger.info("Save Textual links")
        store_dataframe(train_textual_links, "train_textual_links.parquet", dataset_name, object_to_annotate, random_state)
        store_dataframe(test_textual_links, "test_textual_links.parquet", dataset_name, object_to_annotate, random_state)

        






