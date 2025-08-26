import pandas as pd
import logging

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Process metadata2kg-round1 data")

    # load r1 glossary and save in csv format
    raw_r1_glossary_path = "gold_data/raw_data/metadata2kg/round1/r1_glossary.jsonl"
    raw_r1_glossary_df = pd.read_json(raw_r1_glossary_path, lines=True)
    raw_r1_glossary_df.to_csv("gold_data/processed_data/metadata2kg_round1/business_glossary/glossary.csv")

    # load r1 test metadata
    raw_r1_test_metadata_path = "gold_data/raw_data/metadata2kg/round1/r1_test_metadata.jsonl"
    raw_r1_test_metadata_df = pd.read_json(raw_r1_test_metadata_path, lines=True)

    # load r1 test metadata ground truth
    raw_r1_test_gt_path = "gold_data/raw_data/metadata2kg/round1/r1_test_metadata_GT.csv"
    raw_r1_test_gt_df = pd.read_csv(raw_r1_test_gt_path)
    raw_r1_test_gt_df.columns = ['column_id', 'entity_id']

    # generate alignments and save in csv format
    column_alignments_r1 = pd.merge(raw_r1_test_metadata_df, raw_r1_test_gt_df, left_on='id', right_on='column_id', how='inner')
    column_alignments_r1.to_csv("gold_data/processed_data/metadata2kg_round1/alignments/column_alignments/column_to_business_glossary_alignments.csv")

    logger.info("Process metadata2kg-round2 data")

    # load r2 glossary and save in csv format
    raw_r2_glossary_path = "gold_data/raw_data/metadata2kg/round2/r2_glossary.jsonl"
    raw_r2_glossary_df = pd.read_json(raw_r2_glossary_path, lines=True)
    raw_r2_glossary_df.to_csv("gold_data/processed_data/metadata2kg_round2/business_glossary/glossary.csv")

    # load r2 test metadata
    raw_r2_test_metadata_path = "gold_data/raw_data/metadata2kg/round2/r2_test_metadata.jsonl"
    raw_r2_test_metadata_df = pd.read_json(raw_r2_test_metadata_path, lines=True)

    # load r2 test metadata ground truth
    raw_r2_test_gt_path = "gold_data/raw_data/metadata2kg/round2/r2_test_metadata_GT.csv"
    raw_r2_test_gt_df = pd.read_csv(raw_r2_test_gt_path)
    raw_r2_test_gt_df.columns = ['column_id', 'entity_id']

    # generate alignments and save in csv format
    column_alignments_r2 = pd.merge(raw_r2_test_metadata_df, raw_r2_test_gt_df, left_on='id', right_on='column_id', how='inner')
    column_alignments_r2.to_csv("gold_data/processed_data/metadata2kg_round2/alignments/column_alignments/column_to_business_glossary_alignments.csv")

