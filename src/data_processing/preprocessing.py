"""
Data preprocessing module for heart disease dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    """Data preprocessing class for heart disease dataset"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.preprocessor = None
        self._is_fitted = False

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        # Fill numerical columns with median
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numerical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        return df_clean

    def convert_target_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert target column to binary (0/1)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with binary target
        """
        df_clean = df.copy()
        if 'target' in df_clean.columns:
            # Ensure target is binary
            df_clean['target'] = df_clean['target'].apply(
                lambda x: 1 if x > 0 else 0 if pd.notna(x) else 0
            )
        return df_clean

    def scale_features(self, df: pd.DataFrame, numerical_features: List[str]) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_features (List[str]): List of numerical feature names
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_clean = df.copy()
        
        # Only scale features that exist in the dataframe
        existing_features = [col for col in numerical_features if col in df_clean.columns]
        
        if existing_features:
            # Fit and transform
            scaled_data = self.scaler.fit_transform(df_clean[existing_features])
            scaled_df = pd.DataFrame(
                scaled_data, 
                columns=[f"{col}_scaled" for col in existing_features],
                index=df_clean.index
            )
            df_clean = pd.concat([df_clean, scaled_df], axis=1)
        
        return df_clean

    def encode_categorical(self, df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_features (List[str]): List of categorical feature names
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df_clean = df.copy()
        
        # Only encode features that exist in the dataframe
        existing_features = [col for col in categorical_features if col in df_clean.columns]
        
        if existing_features:
            # Fit and transform
            encoded_data = self.encoder.fit_transform(df_clean[existing_features])
            encoded_cols = self.encoder.get_feature_names_out(existing_features)
            encoded_df = pd.DataFrame(
                encoded_data, 
                columns=encoded_cols,
                index=df_clean.index
            )
            df_clean = pd.concat([df_clean, encoded_df], axis=1)
        
        return df_clean

    def preprocess(self, df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            config (Dict): Configuration dictionary
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Convert target to binary
        df_clean = self.convert_target_to_binary(df_clean)
        
        # Extract features and target
        numerical_features = config.get('numerical_features', [])
        categorical_features = config.get('categorical_features', [])
        target_col = config.get('target', 'target')
        
        # Ensure columns exist
        numerical_features = [col for col in numerical_features if col in df_clean.columns]
        categorical_features = [col for col in categorical_features if col in df_clean.columns]
        
        # Check if we have both feature types
        if numerical_features or categorical_features:
            # Create transformers
            transformers = []
            
            if numerical_features:
                transformers.append(('num', StandardScaler(), numerical_features))
            
            if categorical_features:
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                                   categorical_features))
            
            # Create and fit preprocessor
            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
            
            # Fit and transform
            X = df_clean[numerical_features + categorical_features]
            y = df_clean[target_col]
            
            X_transformed = self.preprocessor.fit_transform(X)
            self._is_fitted = True
            
            # Get feature names
            feature_names = []
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    # Get feature names from fitted encoder
                    encoder = transformer
                    if hasattr(encoder, 'get_feature_names_out'):
                        encoded_names = encoder.get_feature_names_out(features)
                        feature_names.extend(encoded_names)
                    else:
                        # Fallback for older sklearn versions
                        for feature in features:
                            unique_vals = df_clean[feature].dropna().unique()
                            for val in sorted(unique_vals)[1:]:  # Skip first due to drop='first'
                                feature_names.append(f"{feature}_{val}")
            
            X_df = pd.DataFrame(X_transformed, columns=feature_names, index=df_clean.index)
            
            return X_df, y
        
        else:
            # Return original features if no preprocessing needed
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]
            return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self._is_fitted or self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        return self.preprocessor.transform(df)
