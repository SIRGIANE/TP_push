"""Module pour les prédictions cliniques"""
from abc import ABC, abstractmethod


class Model(ABC):
    """Classe de base abstraite pour tous les modèles"""
    
    @abstractmethod
    def train(self, X, y):
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Fait des prédictions"""
        pass
    
    def predict_proba(self, X):
        """Retourne les probabilités (optionnel)"""
        raise NotImplementedError("predict_proba non implémenté pour ce modèle")


class ClinicalPredictor:
    """Classe pour effectuer des diagnostics à partir d'un modèle"""
    
    def __init__(self, model):
        self.model = model
        
    def diagnose(self, patient_data):
        """
        Effectue un diagnostic sur les données d'un patient
        
        Args:
            patient_data: Données du patient (array-like)
            
        Returns:
            str: "infecté" si probabilité >= 0.5, "sain" sinon
        """
        prob = self.model.predict_proba(patient_data)[0][1]  # probabilité d'être malade
        return "infecté" if prob >= 0.5 else "sain"
    
    def get_probability(self, patient_data):
        """
        Retourne la probabilité d'être malade
        
        Args:
            patient_data: Données du patient
            
        Returns:
            float: Probabilité d'être malade (classe 1)
        """
        return self.model.predict_proba(patient_data)[0][1]
    
    def diagnose_batch(self, patients_data):
        """
        Effectue des diagnostics sur plusieurs patients
        
        Args:
            patients_data: Données de plusieurs patients
            
        Returns:
            list: Liste de diagnostics
        """
        probs = self.model.predict_proba(patients_data)[:, 1]
        return ["infecté" if p >= 0.5 else "sain" for p in probs]