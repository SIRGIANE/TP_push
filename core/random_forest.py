"""Module pour le modèle Random Forest"""
from sklearn.ensemble import RandomForestClassifier
import joblib
from .model import Model

#SIRGIANE OUIÇAL
class RandomForestModel(Model):
    """Wrapper pour RandomForestClassifier (classification binaire)"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        return self