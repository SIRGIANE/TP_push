"""Module pour le modèle de régression logistique"""
from sklearn.linear_model import LogisticRegression
import joblib
from .model import Model


class LogisticRegressionModel(Model):
    """Classe wrapper pour la régression logistique"""
    
    def __init__(self, max_iter=5000, **kwargs):
        self.model = LogisticRegression(max_iter=max_iter, **kwargs)
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Fait des prédictions"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Retourne les probabilités de prédiction"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Charge un modèle sauvegardé"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        return self