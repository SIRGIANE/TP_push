"""Module pour l'évaluation des modèles"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    """Classe pour évaluer les performances des modèles"""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur l'ensemble de test
        
        Returns:
            dict: Dictionnaire contenant les métriques
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def print_metrics(self, X_test, y_test):
        """Affiche les métriques d'évaluation"""
        metrics = self.evaluate(X_test, y_test)
        
        print("=" * 50)
        print("ÉVALUATION DU MODÈLE")
        print("=" * 50)
        print(f"Précision (Accuracy): {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("=" * 50)
        
        return metrics