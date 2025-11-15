"""Module pour l'entraînement des modèles"""
from core.dataset import Dataset
from core.logisticregression import LogisticRegressionModel
from core.neural_network import NeuralNetwork
from core.loss import LossCalculator


class Trainer:
    """Classe pour gérer l'entraînement des modèles"""
    
    def __init__(self, model_class=LogisticRegressionModel, model_params=None):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = None
        self.dataset = Dataset()
        self.loss_calculator = None
        self.training_history = {
            'train_losses': {},
            'test_losses': {}
        }

    def _initialize_model(self, X_shape):
        """
        Initialise le modèle avec les paramètres appropriés selon son type
        
        Args:
            X_shape: Shape des données d'entrée pour déterminer input_size
        """
        if self.model_class == NeuralNetwork:
            # Paramètres par défaut pour le réseau de neurones si non spécifiés
            default_params = {
                'input_size': X_shape[1],  # Nombre de features
                'hidden_size': 16,         # Taille de la couche cachée par défaut
                'output_size': 1           # Classification binaire
            }
            # Fusionner avec les paramètres fournis par l'utilisateur
            self.model_params = {**default_params, **self.model_params}
            
        return self.model_class(**self.model_params)

    def train_model(self, X_train, y_train, **model_params):
        """Entraîne le modèle"""
        # Mettre à jour les paramètres du modèle avec ceux fournis
        self.model_params.update(model_params)
        
        # Initialiser le modèle avec les bonnes dimensions
        self.model = self._initialize_model(X_train.shape)
        
        # Entraîner le modèle
        self.model.train(X_train, y_train)
        
        # Initialiser le calculator de loss avec le modèle entraîné
        self.loss_calculator = LossCalculator(self.model)
        
        return self.model
    
    def calculate_losses(self, X_train, y_train, X_test, y_test):
        """
        Calcule les pertes sur les ensembles train et test
        
        Args:
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            
        Returns:
            dict: Pertes train et test
        """
        if self.loss_calculator is None:
            raise ValueError("Le modèle doit être entraîné avant de calculer les pertes")
        
        comparison = self.loss_calculator.compare_train_test_loss(
            X_train, y_train, X_test, y_test
        )
        
        # Sauvegarder dans l'historique
        self.training_history['train_losses'] = comparison['train']
        self.training_history['test_losses'] = comparison['test']
        
        return comparison
    
    def run_training(self, test_size=0.2, random_state=42, 
                     calculate_loss=True, verbose=True, **model_params):
        """
        Exécute le pipeline complet d'entraînement
        
        Args:
            test_size: Taille de l'ensemble de test
            random_state: Seed pour la reproductibilité
            calculate_loss: Si True, calcule les pertes après entraînement
            verbose: Si True, affiche les informations
            **model_params: Paramètres du modèle
            
        Returns:
            tuple: (model, X_test, y_test) ou (model, X_test, y_test, losses)
        """
        if verbose:
            print("Préparation des données...")
        
        # Utiliser la méthode split_data de Dataset
        X_train, X_test, y_train, y_test = self.dataset.split_data(
            test_size=test_size,
            random_state=random_state
        )
        
        if verbose:
            print(f"   ✓ Train: {X_train.shape[0]} échantillons")
            print(f"   ✓ Test: {X_test.shape[0]} échantillons")
            print("\n🤖 Entraînement du modèle...")
        
        model = self.train_model(X_train, y_train, **model_params)
        
        if verbose:
            print("   ✓ Modèle entraîné avec succès!")
        
        # Calculer les pertes si demandé
        if calculate_loss:
            if verbose:
                print("\n📈 Calcul des pertes...")
            
            losses = self.calculate_losses(X_train, y_train, X_test, y_test)
            
            if verbose:
                print(f"   ✓ Log Loss (train): {losses['train'].get('log_loss', 'N/A'):.4f}")
                print(f"   ✓ Log Loss (test): {losses['test'].get('log_loss', 'N/A'):.4f}")
            
            return model, X_test, y_test, losses
        
        return model, X_test, y_test
    
    def run_training_with_report(self, test_size=0.2, random_state=42, **model_params):
        """
        Exécute l'entraînement avec un rapport détaillé
        
        Returns:
            tuple: (model, X_test, y_test)
        """
        print("="*60)
        print(" ENTRAÎNEMENT DU MODÈLE")
        print("="*60)
        
        # Entraîner avec calcul de loss
        result = self.run_training(
            test_size=test_size,
            random_state=random_state,
            calculate_loss=True,
            verbose=True,
            **model_params
        )
        
        model, X_test, y_test, losses = result
        
        # Afficher le rapport détaillé des pertes
        print("\n" + "="*60)
        print(" RAPPORT DES PERTES")
        print("="*60)
        self.loss_calculator.print_train_test_comparison(
            self.dataset.X_train, 
            self.dataset.y_train, 
            X_test, 
            y_test
        )
        
        return model, X_test, y_test
    
    def get_training_history(self):
        """
        Retourne l'historique d'entraînement
        
        Returns:
            dict: Historique des pertes
        """
        return self.training_history
    
    def save_model(self, filepath="model.pkl"):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            filepath: Chemin de sauvegarde
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        self.model.save(filepath)
        print(f"✓ Modèle sauvegardé dans '{filepath}'")
    
    def print_summary(self):
        """Affiche un résumé de l'entraînement"""
        if not self.training_history['train_losses']:
            print("Aucun historique d'entraînement disponible")
            return
        
        print("\n" + "="*60)
        print(" RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("="*60)
        
        print("\nPertes d'entraînement:")
        for key, value in self.training_history['train_losses'].items():
            print(f"  {key.upper()}: {value:.6f}")
        
        print("\nPertes de test:")
        for key, value in self.training_history['test_losses'].items():
            print(f"  {key.upper()}: {value:.6f}")
        
        print("="*60)


class TrainerWithValidation(Trainer):
    """Trainer avec validation supplémentaire"""
    
    def run_training_with_validation(self, test_size=0.2, val_size=0.1, 
                                    random_state=42, **model_params):
        """
        Entraîne avec ensemble de validation
        
        Args:
            test_size: Taille de l'ensemble de test
            val_size: Taille de l'ensemble de validation (pris sur train)
            random_state: Seed
            **model_params: Paramètres du modèle
            
        Returns:
            tuple: (model, X_val, X_test, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        print("="*60)
        print(" ENTRAÎNEMENT AVEC VALIDATION")
        print("="*60)
        
        # Utiliser split_data de Dataset
        X_train, X_test, y_train, y_test = self.dataset.split_data(
            test_size=test_size,
            random_state=random_state
        )
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=random_state
        )
        
        print(f"\n Données préparées:")
        print(f"   Train: {X_train.shape[0]} échantillons")
        print(f"   Validation: {X_val.shape[0]} échantillons")
        print(f"   Test: {X_test.shape[0]} échantillons")
        
        # Entraîner
        print("\n Entraînement en cours...")
        model = self.train_model(X_train, y_train, **model_params)
        print("   ✓ Modèle entraîné!")
        
        # Évaluer sur validation
        print("\n Évaluation sur validation:")
        val_losses = self.loss_calculator.evaluate_model_loss(X_val, y_val)
        self.loss_calculator.print_losses(val_losses, "PERTES VALIDATION")
        
        # Évaluer sur test
        print("\n Évaluation sur test:")
        test_losses = self.loss_calculator.evaluate_model_loss(X_test, y_test)
        self.loss_calculator.print_losses(test_losses, "PERTES TEST")
        
        return model, X_val, X_test, y_val, y_test