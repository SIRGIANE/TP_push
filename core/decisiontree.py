"""Module pour le modèle d'arbre de décision"""
from sklearn.tree import DecisionTreeClassifier
import joblib
from .model import Model

class DecisionTreeModel(Model):
    """Classe wrapper pour l'arbre de décision"""
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
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
