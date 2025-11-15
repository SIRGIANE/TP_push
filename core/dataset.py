"""Module pour la gestion des datasets"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


class Dataset:
    """Classe pour gérer le chargement et la division des données"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Charge le dataset breast cancer"""
        self.data = load_breast_cancer()
        return self.data
    
    def split_data(self, test_size=0.2, random_state=42):
        """Divise les données en ensembles d'entraînement et de test"""
        if self.data is None:
            self.load_data()
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.data, 
            self.data.target, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def to_dataframe(self):
        """Convertit les données en DataFrame pandas"""
        if self.data is None:
            self.load_data()
            
        df = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        df["diagnostic"] = self.data.target
        return df
    
    def get_feature_names(self):
        """Retourne les noms des features"""
        if self.data is None:
            self.load_data()
        return self.data.feature_names
    
    def get_target_names(self):
        """Retourne les noms des classes cibles"""
        if self.data is None:
            self.load_data()
        return self.data.target_names