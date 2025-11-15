"""Module pour le calcul et l'analyse des fonctions de perte"""
from sklearn.metrics import (
    log_loss, 
    mean_squared_error, 
    mean_absolute_error,
    hinge_loss
)
import numpy as np


class LossCalculator:
    """Classe pour calculer et analyser les fonctions de perte"""
    
    def __init__(self, model=None):
        """
        Args:
            model: Modèle sklearn (optionnel)
        """
        self.model = model
    
    def calculate_log_loss(self, y_true, y_pred_proba):
        """
        Calcule la log loss (Binary Cross-Entropy)
        
        Args:
            y_true: Valeurs réelles
            y_pred_proba: Probabilités prédites
            
        Returns:
            float: Log loss
        """
        return log_loss(y_true, y_pred_proba)
    
    def calculate_mse(self, y_true, y_pred):
        """
        Calcule le Mean Squared Error
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            float: MSE
        """
        return mean_squared_error(y_true, y_pred)
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calcule le Mean Absolute Error
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            float: MAE
        """
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_hinge_loss(self, y_true, y_pred):
        """
        Calcule la Hinge Loss
        
        Args:
            y_true: Valeurs réelles
            y_pred: Décisions prédites
            
        Returns:
            float: Hinge loss
        """
        return hinge_loss(y_true, y_pred)
    
    def calculate_all_losses(self, y_true, y_pred, y_pred_proba=None):
        """
        Calcule toutes les pertes disponibles
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions (classes)
            y_pred_proba: Probabilités prédites (optionnel)
            
        Returns:
            dict: Dictionnaire avec toutes les pertes
        """
        losses = {}
        
        # MSE et MAE (toujours calculables)
        losses['mse'] = self.calculate_mse(y_true, y_pred)
        losses['mae'] = self.calculate_mae(y_true, y_pred)
        losses['rmse'] = np.sqrt(losses['mse'])
        
        # Log loss (nécessite les probabilités)
        if y_pred_proba is not None:
            losses['log_loss'] = self.calculate_log_loss(y_true, y_pred_proba)
        
        return losses
    
    def evaluate_model_loss(self, X, y):
        """
        Évalue la perte du modèle sur un ensemble de données
        
        Args:
            X: Features
            y: Target
            
        Returns:
            dict: Pertes calculées
        """
        if self.model is None:
            raise ValueError("Aucun modèle n'a été fourni")
        
        # Prédictions
        y_pred = self.model.predict(X)
        
        # Probabilités (si disponible)
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        return self.calculate_all_losses(y, y_pred, y_pred_proba)
    
    def compare_train_test_loss(self, X_train, y_train, X_test, y_test):
        """
        Compare les pertes sur train et test
        
        Args:
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            
        Returns:
            dict: Pertes train et test
        """
        if self.model is None:
            raise ValueError("Aucun modèle n'a été fourni")
        
        train_losses = self.evaluate_model_loss(X_train, y_train)
        test_losses = self.evaluate_model_loss(X_test, y_test)
        
        return {
            'train': train_losses,
            'test': test_losses,
            'difference': {
                key: test_losses[key] - train_losses[key]
                for key in train_losses.keys()
            }
        }
    
    def print_losses(self, losses, title="FONCTIONS DE PERTE"):
        """
        Affiche les pertes de manière formatée
        
        Args:
            losses: Dictionnaire de pertes
            title: Titre à afficher
        """
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)
        
        for name, value in losses.items():
            if isinstance(value, dict):
                print(f"\n{name.upper()}:")
                for sub_name, sub_value in value.items():
                    print(f"  {sub_name:.<25} {sub_value:.6f}")
            else:
                print(f"{name.upper():.<30} {value:.6f}")
        
        print("="*60)
    
    def print_train_test_comparison(self, X_train, y_train, X_test, y_test):
        """
        Affiche une comparaison formatée des pertes train/test
        
        Args:
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
        """
        comparison = self.compare_train_test_loss(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print(" COMPARAISON TRAIN vs TEST")
        print("="*60)
        
        # En-tête
        print(f"\n{'Métrique':<20} {'Train':<15} {'Test':<15} {'Différence':<15}")
        print("-"*60)
        
        # Données
        for key in comparison['train'].keys():
            train_val = comparison['train'][key]
            test_val = comparison['test'][key]
            diff = comparison['difference'][key]
            
            print(f"{key.upper():<20} {train_val:<15.6f} {test_val:<15.6f} {diff:<15.6f}")
        
        # Analyse
        print("\n" + "-"*60)
        log_loss_diff = comparison['difference'].get('log_loss', 0)
        
        if log_loss_diff > 0.1:
            print("⚠️  ATTENTION: Possible overfitting détecté!")
            print(f"   La log loss de test est {log_loss_diff:.4f} plus élevée que train")
        elif log_loss_diff < -0.05:
            print("⚠️  ATTENTION: Possible underfitting détecté!")
        else:
            print("✓  Bonne généralisation du modèle")
        
        print("="*60)


def quick_loss_evaluation(model, X_test, y_test):
    """
    Fonction rapide pour évaluer les pertes d'un modèle
    
    Args:
        model: Modèle sklearn
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        dict: Pertes calculées
    """
    calculator = LossCalculator(model)
    return calculator.evaluate_model_loss(X_test, y_test)


def print_quick_loss(model, X_test, y_test):
    """
    Affiche rapidement les pertes d'un modèle
    
    Args:
        model: Modèle sklearn
        X_test: Features de test
        y_test: Target de test
    """
    calculator = LossCalculator(model)
    losses = calculator.evaluate_model_loss(X_test, y_test)
    calculator.print_losses(losses, "ÉVALUATION DES PERTES")