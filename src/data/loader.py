"""
Data loading module for consumer complaints dataset
Loads data from BigQuery and prepares it for training
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare consumer complaints data from BigQuery"""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str = "consumer_complaints",
        table_id: str = "complaints"
    ):
        """
        Initialize data loader
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=project_id)
        
        logger.info(f"DataLoader initialized for {project_id}.{dataset_id}.{table_id}")
    
    def load_data(
        self,
        target_column: str = "Product",
        limit: Optional[int] = None,
        min_text_length: int = 50,
        max_text_length: int = 10000
    ) -> pd.DataFrame:
        """
        Load data from BigQuery
        
        Args:
            target_column: Column to use as prediction target
            limit: Maximum number of rows to load (None for all)
            min_text_length: Minimum complaint text length
            max_text_length: Maximum complaint text length
            
        Returns:
            DataFrame with cleaned data
        """
        logger.info(f"Loading data from BigQuery...")
        logger.info(f"  Target: {target_column}")
        logger.info(f"  Limit: {limit if limit else 'None (all data)'}")
        
        # Build query
        query = f"""
        SELECT 
            Consumer_Complaint as text,
            {target_column} as target,
            State as state,
            Company as company,
            Company_Response_to_Consumer as response
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        WHERE 
            Consumer_Complaint IS NOT NULL 
            AND {target_column} IS NOT NULL
            AND LENGTH(Consumer_Complaint) >= {min_text_length}
            AND LENGTH(Consumer_Complaint) <= {max_text_length}
        """
        
        if limit:
            query += f"\nLIMIT {limit}"
        
        # Execute query
        logger.info("Executing BigQuery query...")
        df = self.client.query(query).to_dataframe()
        
        logger.info(f"✅ Loaded {len(df):,} rows")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Unique targets: {df['target'].nunique()}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean complaint text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test...")
        
        # Clean text
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove empty text
        df = df[df['text'].str.len() > 0]
        
        # Encode labels (will be done by LabelEncoder later, but check uniqueness)
        unique_targets = df['target'].nunique()
        logger.info(f"  Unique targets: {unique_targets}")
        
        # Split train/test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['target']
        )
        
        # Split train/val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_df['target']
        )
        
        logger.info(f"✅ Data split complete:")
        logger.info(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Get class distribution statistics"""
        return df['target'].value_counts()
    
    def compute_class_imbalance(self, df: pd.DataFrame) -> float:
        """Compute class imbalance ratio"""
        counts = df['target'].value_counts()
        return counts.max() / counts.min()


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python loader.py PROJECT_ID")
        sys.exit(1)
    
    project_id = sys.argv[1]
    
    loader = DataLoader(project_id)
    df = loader.load_data(limit=1000)
    
    print(f"\nDataset summary:")
    print(f"  Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(loader.get_class_distribution(df))
    print(f"\nImbalance ratio: {loader.compute_class_imbalance(df):.2f}:1")
