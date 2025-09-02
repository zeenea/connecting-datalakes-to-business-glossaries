import pandas as pd
import logging


def process_metadata2kg_data():
    logger.info("Process metadata2kg-round1 data")
    logger.info("Load glossary")
    # load r1 glossary and save in csv format
    raw_r1_glossary_path = "gold_data/raw_data/metadata2kg/round1/r1_glossary.jsonl"
    raw_r1_glossary_df = pd.read_json(raw_r1_glossary_path, lines=True)
    raw_r1_glossary_df.to_csv("gold_data/processed_data/metadata2kg_round1/business_glossary/glossary.csv")

    logger.info("Load test metadata and ground truth")
    # load r1 test metadata
    raw_r1_test_metadata_path = "gold_data/raw_data/metadata2kg/round1/r1_test_metadata.jsonl"
    raw_r1_test_metadata_df = pd.read_json(raw_r1_test_metadata_path, lines=True)

    # load r1 test metadata ground truth
    raw_r1_test_gt_path = "gold_data/raw_data/metadata2kg/round1/r1_test_metadata_GT.csv"
    raw_r1_test_gt_df = pd.read_csv(raw_r1_test_gt_path)
    raw_r1_test_gt_df.columns = ['column_id', 'entity_id']

    logger.info("Generate test column alignments file")
    # generate test alignments and save in csv format
    test_column_alignments_r1 = pd.merge(raw_r1_test_metadata_df, raw_r1_test_gt_df, left_on='id', right_on='column_id', how='inner')
    test_column_alignments_r1 = pd.merge(test_column_alignments_r1, raw_r1_glossary_df[['id', 'label']].rename(columns={'id':'entity_id', 'label':'entity_label'}), left_on='entity_id', right_on='entity_id', how='inner')
    test_column_alignments_r1.to_csv("gold_data/processed_data/metadata2kg_round1/alignments/column_alignments/test_column_to_business_glossary_alignments.csv")

    logger.info("Load eval metadata and ground truth")
    # load r1 sample metadata
    raw_r1_eval_metadata_path = "gold_data/raw_data/metadata2kg/round1/r1_sample_metadata.jsonl"
    raw_r1_eval_metadata_df = pd.read_json(raw_r1_eval_metadata_path, lines=True)

    # load r1 sample ground truth
    raw_r1_eval_gt_path = "gold_data/raw_data/metadata2kg/round1/r1_sample_GT.csv"
    raw_r1_eval_gt_df = pd.read_csv(raw_r1_eval_gt_path)
    raw_r1_eval_gt_df.columns = ['column_id', 'entity_id']

    logger.info("Generate eval column alignments file")
    # generate sample alignments and save in csv format
    eval_column_alignments_r1 = pd.merge(raw_r1_eval_metadata_df, raw_r1_eval_gt_df, left_on='id', right_on='column_id', how='inner')
    eval_column_alignments_r1 = pd.merge(eval_column_alignments_r1, raw_r1_glossary_df[['id', 'label']].rename(columns={'id':'entity_id', 'label':'entity_label'}), left_on='entity_id', right_on='entity_id', how='inner')
    eval_column_alignments_r1.to_csv("gold_data/processed_data/metadata2kg_round1/alignments/column_alignments/eval_column_to_business_glossary_alignments.csv")


    logger.info("Process metadata2kg-round2 data")
    logger.info("Load glossary")
    # load r2 glossary and save in csv format
    raw_r2_glossary_path = "gold_data/raw_data/metadata2kg/round2/r2_glossary.jsonl"
    raw_r2_glossary_df = pd.read_json(raw_r2_glossary_path, lines=True)
    raw_r2_glossary_df.to_csv("gold_data/processed_data/metadata2kg_round2/business_glossary/glossary.csv")

    logger.info("Load test metadata and ground truth")
    # load r2 test metadata
    raw_r2_test_metadata_path = "gold_data/raw_data/metadata2kg/round2/r2_test_metadata.jsonl"
    raw_r2_test_metadata_df = pd.read_json(raw_r2_test_metadata_path, lines=True)

    # load r2 test metadata ground truth
    raw_r2_test_gt_path = "gold_data/raw_data/metadata2kg/round2/r2_test_metadata_GT.csv"
    raw_r2_test_gt_df = pd.read_csv(raw_r2_test_gt_path)
    raw_r2_test_gt_df.columns = ['column_id', 'entity_id']

    logger.info("Generate test column alignments file")
    # generate alignments and save in csv format
    test_column_alignments_r2 = pd.merge(raw_r2_test_metadata_df, raw_r2_test_gt_df, left_on='id', right_on='column_id', how='inner')
    test_column_alignments_r2 = pd.merge(test_column_alignments_r2, raw_r2_glossary_df[['id', 'label']].rename(columns={'id':'entity_id', 'label':'entity_label'}), left_on='entity_id', right_on='entity_id', how='inner')
    test_column_alignments_r2.to_csv("gold_data/processed_data/metadata2kg_round2/alignments/column_alignments/test_column_to_business_glossary_alignments.csv")

    logger.info("Load eval metadata and ground truth")
    # load r2 sample metadata
    raw_r2_eval_metadata_path = "gold_data/raw_data/metadata2kg/round2/r2_sample_metadata.jsonl"
    raw_r2_eval_metadata_df = pd.read_json(raw_r2_eval_metadata_path, lines=True)

    # load r2 sample metadata ground truth
    raw_r2_eval_gt_path = "gold_data/raw_data/metadata2kg/round2/r2_sample_metadata_GT.csv"
    raw_r2_eval_gt_df = pd.read_csv(raw_r2_eval_gt_path)
    raw_r2_eval_gt_df.columns = ['column_id', 'entity_id']

    logger.info("Generate eval column alignments file")
    # generate alignments and save in csv format
    eval_column_alignments_r2 = pd.merge(raw_r2_eval_metadata_df, raw_r2_eval_gt_df, left_on='id', right_on='column_id', how='inner')
    eval_column_alignments_r2 = pd.merge(eval_column_alignments_r2, raw_r2_glossary_df[['id', 'label']].rename(columns={'id':'entity_id', 'label':'entity_label'}), left_on='entity_id', right_on='entity_id', how='inner')
    eval_column_alignments_r2.to_csv("gold_data/processed_data/metadata2kg_round2/alignments/column_alignments/eval_column_to_business_glossary_alignments.csv")


def process_zeenea_open_ds_data():

    logger.info("Add table headers to dataset alignments")

    table_alignments_path = "../gold_data/raw_data/zeenea-open-ds/alignments/dataset_to_business_glossary_alignments.csv"
    table_alignments = pd.read_csv(table_alignments_path, index_col=0)

    headers = []

    for idx, row in table_alignments.iterrows():

        table_path = f"../gold_data/raw_data/zeenea-open-ds/data/{row['dataset_code']}.csv"
        table = pd.read_csv(table_path, index_col=0)
        header = list(table.columns)
        headers.append(header)

    table_alignments['table_columns'] = headers

    table_alignments.to_csv("../gold_data/processed_data/zeenea-open-ds/alignments/dataset_to_business_glossary_alignments.csv")




if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    #process_metadata2kg_data()

    #process_zeenea_open_ds_data()
