"""Interface interactive pour choisir et utiliser différents modèles"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipline.trainer import Trainer
from pipline.evaluator import Evaluator
from core.logisticregression import LogisticRegressionModel
from core.neural_network import NeuralNetwork
from core.dataset import Dataset
from utils.metrics import MetricsCalculator
import numpy as np
import pickle


class ModelInterface:
    """Interface interactive pour la sélection et l'utilisation de modèles"""
    
    def __init__(self):
        self.available_models = {
            '1': {
                'name': 'Régression Logistique',
                'class': LogisticRegressionModel,
                'default_params': {'max_iter': 5000}
            },
            '2': {
                'name': 'Réseau de Neurones',
                'class': NeuralNetwork,
                'default_params': {'hidden_size': 16, 'epochs': 1000, 'learning_rate': 0.01}
            }
        }
        self.current_model = None
        self.current_model_type = None
        self.trained_model = None
        self.X_test = None
        self.y_test = None
        self.dataset = Dataset()
        
    def clear_screen(self):
        """Efface l'écran"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_main_menu(self):
        """Affiche le menu principal"""
        print("\n" + "="*70)
        print(" 🤖 SYSTÈME DE PRÉDICTION DE CANCER - SÉLECTION DE MODÈLE")
        print("="*70)
        print("\n📋 MENU PRINCIPAL:")
        print("\n1. Sélectionner et entraîner un modèle")
        print("2. Faire une prédiction (modèle entraîné requis)")
        print("3. Évaluer le modèle actuel")
        print("4. Comparer les modèles")
        print("5. Charger un modèle sauvegardé")
        print("6. Sauvegarder le modèle actuel")
        print("7. Afficher les informations sur les modèles")
        print("8. Quitter")
        print("-"*70)
        
        if self.current_model:
            print(f"\n✓ Modèle actuel: {self.current_model_type}")
        else:
            print("\n⚠️  Aucun modèle chargé")
        
        print("-"*70)
    
    def display_model_selection(self):
        """Affiche le menu de sélection de modèle"""
        print("\n" + "="*70)
        print(" 📊 SÉLECTION DU MODÈLE")
        print("="*70)
        print("\nModèles disponibles:\n")
        
        for key, model_info in self.available_models.items():
            print(f"{key}. {model_info['name']}")
            print(f"   Paramètres par défaut: {model_info['default_params']}")
            print()
        
        print("0. Retour au menu principal")
        print("-"*70)
    
    def select_and_train_model(self):
        """Permet à l'utilisateur de sélectionner et entraîner un modèle"""
        self.display_model_selection()
        
        choice = input("\nChoisissez un modèle (0-2): ").strip()
        
        if choice == '0':
            return
        
        if choice not in self.available_models:
            print("\n❌ Choix invalide!")
            input("Appuyez sur Entrée pour continuer...")
            return
        
        model_info = self.available_models[choice]
        self.current_model_type = model_info['name']
        
        print(f"\n✓ Modèle sélectionné: {model_info['name']}")
        
        # Demander les paramètres personnalisés
        print("\n📝 Configuration des paramètres:")
        print("(Appuyez sur Entrée pour utiliser les valeurs par défaut)")
        
        custom_params = {}
        
        if choice == '1':  # Logistic Regression
            max_iter = input(f"Nombre d'itérations (défaut: 5000): ").strip()
            if max_iter:
                custom_params['max_iter'] = int(max_iter)
        
        elif choice == '2':  # Neural Network
            hidden_size = input(f"Taille couche cachée (défaut: 16): ").strip()
            if hidden_size:
                custom_params['hidden_size'] = int(hidden_size)
            
            epochs = input(f"Nombre d'époques (défaut: 1000): ").strip()
            if epochs:
                custom_params['epochs'] = int(epochs)
            
            learning_rate = input(f"Taux d'apprentissage (défaut: 0.01): ").strip()
            if learning_rate:
                custom_params['learning_rate'] = float(learning_rate)
        
        # Fusionner avec les paramètres par défaut
        model_params = {**model_info['default_params'], **custom_params}
        
        # Entraîner le modèle
        print("\n" + "="*70)
        print(" 🚀 ENTRAÎNEMENT EN COURS...")
        print("="*70)
        
        try:
            trainer = Trainer(
                model_class=model_info['class'],
                model_params=model_params
            )
            
            self.trained_model, self.X_test, self.y_test, losses = trainer.run_training(
                test_size=0.2,
                random_state=42,
                calculate_loss=True,
                verbose=True
            )
            
            self.current_model = self.trained_model
            
            print("\n✅ Modèle entraîné avec succès!")
            
            # Afficher les métriques
            print("\n" + "="*70)
            print(" 📈 MÉTRIQUES DE PERFORMANCE")
            print("="*70)
            
            evaluator = Evaluator(self.trained_model.model if hasattr(self.trained_model, 'model') else self.trained_model)
            evaluator.print_metrics(self.X_test, self.y_test)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'entraînement: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def make_prediction(self):
        """Faire une prédiction avec le modèle actuel"""
        if not self.current_model:
            print("\n❌ Aucun modèle chargé! Veuillez d'abord entraîner ou charger un modèle.")
            input("Appuyez sur Entrée pour continuer...")
            return
        
        print("\n" + "="*70)
        print(" 🔮 FAIRE UNE PRÉDICTION")
        print("="*70)
        print("\n1. Utiliser des données aléatoires (pour test)")
        print("2. Entrer les données manuellement")
        print("3. Utiliser un échantillon de test")
        print("0. Retour")
        
        choice = input("\nVotre choix: ").strip()
        
        if choice == '0':
            return
        
        elif choice == '1':
            # Générer des données aléatoires
            patient_data = self._generate_random_patient()
            print("\n✓ Données aléatoires générées")
            
        elif choice == '2':
            print("\n⚠️  Saisie manuelle non implémentée dans cette version")
            input("Appuyez sur Entrée pour continuer...")
            return
            
        elif choice == '3':
            if self.X_test is None:
                print("\n❌ Aucune donnée de test disponible!")
                input("Appuyez sur Entrée pour continuer...")
                return
            
            idx = np.random.randint(0, len(self.X_test))
            patient_data = self.X_test[idx]
            print(f"\n✓ Échantillon de test #{idx} sélectionné")
            print(f"   Vraie valeur: {'Malin' if self.y_test[idx] == 1 else 'Bénin'}")
        
        else:
            print("\n❌ Choix invalide!")
            input("Appuyez sur Entrée pour continuer...")
            return
        
        # Faire la prédiction
        try:
            patient_data = patient_data.reshape(1, -1)
            prediction = self.current_model.predict(patient_data)[0]
            
            # Obtenir la probabilité si disponible
            if hasattr(self.current_model, 'predict_proba'):
                proba = self.current_model.predict_proba(patient_data)[0]
                prob_benign = proba[0] if len(proba) > 1 else 1 - proba[0]
                prob_malignant = proba[1] if len(proba) > 1 else proba[0]
            else:
                prob_malignant = prediction
                prob_benign = 1 - prediction
            
            # Afficher le résultat
            print("\n" + "="*70)
            print(" 📊 RÉSULTAT DE LA PRÉDICTION")
            print("="*70)
            print(f"\nModèle utilisé: {self.current_model_type}")
            print(f"\nDiagnostic: {'MALIN ⚠️' if prediction == 1 else 'BÉNIN ✓'}")
            print(f"\nProbabilités:")
            print(f"  • Tumeur bénigne: {prob_benign:.2%}")
            print(f"  • Tumeur maligne: {prob_malignant:.2%}")
            
            if prediction == 1:
                print("\n⚠️  ATTENTION: Résultat positif pour tumeur maligne")
                print("   Recommandation: Consultation médicale immédiate requise")
            else:
                print("\n✓  Résultat: Tumeur bénigne détectée")
                print("   Recommandation: Suivi médical régulier conseillé")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la prédiction: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def evaluate_model(self):
        """Évaluer le modèle actuel"""
        if not self.current_model or self.X_test is None:
            print("\n❌ Aucun modèle entraîné disponible!")
            input("Appuyez sur Entrée pour continuer...")
            return
        
        print("\n" + "="*70)
        print(f" 📊 ÉVALUATION DU MODÈLE: {self.current_model_type}")
        print("="*70)
        
        try:
            # Métriques de base
            evaluator = Evaluator(self.trained_model.model if hasattr(self.trained_model, 'model') else self.trained_model)
            evaluator.print_metrics(self.X_test, self.y_test)
            
            # Métriques détaillées
            print("\n" + "="*70)
            print(" 📈 MÉTRIQUES DÉTAILLÉES")
            print("="*70)
            
            y_pred = self.current_model.predict(self.X_test)
            
            self.dataset.load_data()
            target_names = self.dataset.get_target_names()
            
            metrics_calc = MetricsCalculator()
            metrics_calc.print_confusion_matrix(self.y_test, y_pred, target_names)
            metrics_calc.print_classification_report(self.y_test, y_pred, target_names)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'évaluation: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def compare_models(self):
        """Comparer tous les modèles disponibles"""
        print("\n" + "="*70)
        print(" 🔬 COMPARAISON DES MODÈLES")
        print("="*70)
        print("\nEntraînement de tous les modèles disponibles...")
        
        results = {}
        
        for key, model_info in self.available_models.items():
            print(f"\n{'='*70}")
            print(f" Entraînement: {model_info['name']}")
            print(f"{'='*70}")
            
            try:
                trainer = Trainer(
                    model_class=model_info['class'],
                    model_params=model_info['default_params']
                )
                
                model, X_test, y_test, losses = trainer.run_training(
                    test_size=0.2,
                    random_state=42,
                    calculate_loss=True,
                    verbose=False
                )
                
                # Calculer les métriques
                evaluator = Evaluator(model.model if hasattr(model, 'model') else model)
                metrics = evaluator.calculate_metrics(X_test, y_test)
                
                results[model_info['name']] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'log_loss': losses['test'].get('log_loss', 'N/A')
                }
                
                print(f"✓ {model_info['name']} entraîné avec succès!")
                
            except Exception as e:
                print(f"❌ Erreur avec {model_info['name']}: {str(e)}")
                results[model_info['name']] = None
        
        # Afficher la comparaison
        print("\n" + "="*70)
        print(" 📊 RÉSULTATS DE LA COMPARAISON")
        print("="*70)
        
        print(f"\n{'Modèle':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)
        
        for model_name, metrics in results.items():
            if metrics:
                print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
            else:
                print(f"{model_name:<25} {'Erreur':<12}")
        
        print("="*70)
        
        # Déterminer le meilleur modèle
        if results:
            valid_results = {k: v for k, v in results.items() if v is not None}
            if valid_results:
                best_model = max(valid_results.items(), key=lambda x: x[1]['f1_score'])
                print(f"\n🏆 Meilleur modèle: {best_model[0]}")
                print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def save_model(self):
        """Sauvegarder le modèle actuel"""
        if not self.current_model:
            print("\n❌ Aucun modèle à sauvegarder!")
            input("Appuyez sur Entrée pour continuer...")
            return
        
        filename = input("\nNom du fichier (défaut: model.pkl): ").strip()
        if not filename:
            filename = "model.pkl"
        
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.current_model, f)
            
            print(f"\n✓ Modèle sauvegardé dans '{filename}'")
        except Exception as e:
            print(f"\n❌ Erreur lors de la sauvegarde: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def load_model(self):
        """Charger un modèle sauvegardé"""
        filename = input("\nNom du fichier à charger (défaut: cancer_model.pkl): ").strip()
        if not filename:
            filename = "cancer_model.pkl"
        
        try:
            with open(filename, 'rb') as f:
                self.current_model = pickle.load(f)
            
            self.current_model_type = "Modèle chargé"
            print(f"\n✓ Modèle chargé depuis '{filename}'")
            
            # Charger les données de test si disponibles
            if self.X_test is None:
                self.dataset.load_data()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    self.dataset.X, self.dataset.y, test_size=0.2, random_state=42
                )
                self.X_test = X_test
                self.y_test = y_test
                print("✓ Données de test chargées")
            
        except FileNotFoundError:
            print(f"\n❌ Fichier '{filename}' introuvable!")
        except Exception as e:
            print(f"\n❌ Erreur lors du chargement: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def display_model_info(self):
        """Afficher des informations sur les modèles"""
        print("\n" + "="*70)
        print(" 📚 INFORMATIONS SUR LES MODÈLES")
        print("="*70)
        
        print("\n1. RÉGRESSION LOGISTIQUE")
        print("   " + "-"*65)
        print("   • Type: Modèle linéaire")
        print("   • Principe: Utilise une fonction sigmoïde pour la classification")
        print("   • Avantages: Rapide, interprétable, efficace pour problèmes linéaires")
        print("   • Paramètres: max_iter (nombre d'itérations)")
        
        print("\n2. RÉSEAU DE NEURONES")
        print("   " + "-"*65)
        print("   • Type: Modèle non-linéaire")
        print("   • Architecture: 1 couche cachée avec activation sigmoïde")
        print("   • Avantages: Peut capturer des relations non-linéaires complexes")
        print("   • Paramètres:")
        print("     - hidden_size: Nombre de neurones dans la couche cachée")
        print("     - epochs: Nombre d'époques d'entraînement")
        print("     - learning_rate: Taux d'apprentissage pour la descente de gradient")
        
        print("\n" + "="*70)
        print(" 🎯 CONCEPT POO: POLYMORPHISME")
        print("="*70)
        print("\nTous les modèles implémentent une méthode predict() commune.")
        print("Cela permet de les utiliser de manière interchangeable grâce")
        print("au polymorphisme - un concept clé de la POO!")
        print("="*70)
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def _generate_random_patient(self):
        """Génère des données de patient aléatoires réalistes"""
        return np.array([
            np.random.uniform(6, 28),    # mean radius
            np.random.uniform(9, 40),    # mean texture
            np.random.uniform(40, 190),  # mean perimeter
            np.random.uniform(140, 2500), # mean area
            np.random.uniform(0.05, 0.16), # mean smoothness
            np.random.uniform(0.01, 0.35), # mean compactness
            np.random.uniform(0, 0.43),   # mean concavity
            np.random.uniform(0, 0.2),    # mean concave points
            np.random.uniform(0.1, 0.3),  # mean symmetry
            np.random.uniform(0.04, 0.1), # mean fractal dimension
            np.random.uniform(0.1, 2.9),  # radius error
            np.random.uniform(0.3, 4.9),  # texture error
            np.random.uniform(0.7, 21),   # perimeter error
            np.random.uniform(6, 542),    # area error
            np.random.uniform(0.001, 0.03), # smoothness error
            np.random.uniform(0.002, 0.14), # compactness error
            np.random.uniform(0, 0.4),    # concavity error
            np.random.uniform(0, 0.05),   # concave points error
            np.random.uniform(0.007, 0.08), # symmetry error
            np.random.uniform(0.0008, 0.03), # fractal dimension error
            np.random.uniform(7, 36),     # worst radius
            np.random.uniform(12, 50),    # worst texture
            np.random.uniform(50, 250),   # worst perimeter
            np.random.uniform(185, 4254), # worst area
            np.random.uniform(0.07, 0.22), # worst smoothness
            np.random.uniform(0.02, 1.06), # worst compactness
            np.random.uniform(0, 1.25),   # worst concavity
            np.random.uniform(0, 0.29),   # worst concave points
            np.random.uniform(0.15, 0.66), # worst symmetry
            np.random.uniform(0.05, 0.21)  # worst fractal dimension
        ])
    
    def run(self):
        """Lance l'interface"""
        while True:
            self.display_main_menu()
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == '1':
                self.select_and_train_model()
            elif choice == '2':
                self.make_prediction()
            elif choice == '3':
                self.evaluate_model()
            elif choice == '4':
                self.compare_models()
            elif choice == '5':
                self.load_model()
            elif choice == '6':
                self.save_model()
            elif choice == '7':
                self.display_model_info()
            elif choice == '8':
                print("\n" + "="*70)
                print(" 👋 Merci d'avoir utilisé le système de prédiction!")
                print("="*70)
                print()
                break
            else:
                print("\n❌ Choix invalide. Veuillez réessayer.")
                input("Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    interface = ModelInterface()
    interface.run()
