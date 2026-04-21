import pandas as pd
from typing import Optional
from src.utils.logger import logger

class AnnotationsManager:
    def __init__(self, csv_path: str):
        self.df = None
        self._load_gt_dataframe(csv_path)

    def _load_gt_dataframe(self, csv_path: str):
        logger.info(f"Loading Ground Truth CSV from {csv_path}...")
        try:
            self.df = pd.read_csv(csv_path)
            self.df['start'] = pd.to_numeric(self.df['start'], errors='coerce')
            self.df['end'] = pd.to_numeric(self.df['end'], errors='coerce')
            logger.info(f"GT Loaded: {len(self.df)} rows.")
        except Exception as e:
            logger.error(f"Error loading GT CSV: {e}")
            self.df = pd.DataFrame()

    def get_ground_truth_for_chunk(self, video_stem: str, chunk_index: int, chunk_duration: int, step: int) -> str:
        """
        Filters GT for the specific chunk time window and returns last 15 events.
        """
        if self.df is None or self.df.empty:
            return "No Ground Truth Available"

        start_time = chunk_index * step
        end_time = start_time + chunk_duration 

        video_mask = self.df['video_source'].astype(str).str.contains(video_stem, regex=False, na=False)
        time_mask = (self.df['start'] > start_time) & (self.df['end'] < end_time)

        filtered_df = self.df[video_mask & time_mask].copy()
        filtered_df = filtered_df.sort_values(by='start')

        cols_to_keep = ['subject', 'act', 'utterance_type', 'high_level_action', 'low_level_action', 'target', 'start', 'end']
        existing_cols = [c for c in cols_to_keep if c in filtered_df.columns]
        filtered_df = filtered_df[existing_cols]

        final_rows = filtered_df.iloc[-15:]
        return final_rows.to_json(orient='records')
