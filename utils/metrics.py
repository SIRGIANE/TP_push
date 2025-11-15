"""Module pour les métriques personnalisées"""
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class MetricsCalculator:
    """Classe pour calculer diverses métriques"""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Calcule et retourne la matrice de confusion"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def classification_report_dict(y_true, y_pred, target_names=None):
        """Retourne le rapport de classification sous forme de dictionnaire"""
        return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    @staticmethod
    def print_confusion_matrix(y_true, y_pred, target_names=None):
        """Affiche la matrice de confusion de manière lisible"""
        cm = confusion_matrix(y_true, y_pred)
        
        if target_names is None:
            target_names = ['Classe 0', 'Classe 1']
        
        print("\nMatrice de Confusion:")
        print(f"{'':>15} {'Prédit ' + target_names[0]:>15} {'Prédit ' + target_names[1]:>15}")
        print(f"{'Réel ' + target_names[0]:>15} {cm[0][0]:>15} {cm[0][1]:>15}")
        print(f"{'Réel ' + target_names[1]:>15} {cm[1][0]:>15} {cm[1][1]:>15}")
        
        return cm
    
    @staticmethod
    def print_classification_report(y_true, y_pred, target_names=None):
        """Affiche le rapport de classification"""
        print("\nRapport de Classification:")
        print(classification_report(y_true, y_pred, target_names=target_names))