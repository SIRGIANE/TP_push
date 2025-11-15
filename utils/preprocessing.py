"""Module pour le prétraitement des données"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class DataPreprocessor:
    """Classe pour le prétraitement des données"""
    
    def __init__(self, scaler_type='standard'):
        """
        Args:
            scaler_type: Type de normalisation ('standard' ou 'minmax')
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type doit être 'standard' ou 'minmax'")
        
        self.is_fitted = False
    
    def fit_transform(self, X):
        """Ajuste le scaler et transforme les données"""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X):
        """Transforme les données avec un scaler déjà ajusté"""
        if not self.is_fitted:
            raise ValueError("Le preprocessor doit être ajusté avant la transformation")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        """Inverse la transformation"""
        if not self.is_fitted:
            raise ValueError("Le preprocessor doit être ajusté avant l'inverse transformation")
        return self.scaler.inverse_transform(X)
    
    
    def remove_missing_values(X, y=None):
        """Supprime les valeurs manquantes"""
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        if y is not None:
            y_clean = y[mask]
            return X_clean, y_clean
        
        return X_clean
    
    
    def handle_outliers(X, threshold=3):
        """
        Détecte et gère les outliers basés sur le z-score
        
        Args:
            X: Données
            threshold: Seuil du z-score
            
        Returns:
            Masque booléen des données sans outliers
        """
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        return (z_scores < threshold).all(axis=1)