"""Interface en ligne de commande pour le diagnostic clinique"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import CancerPredictionAPI
import numpy as np


class ClinicalInterface:
    """Interface CLI pour les diagnostics cliniques"""
    
    def __init__(self, model_path="cancer_model.pkl"):
        self.api = CancerPredictionAPI(model_path)
        self.feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
    
    def display_menu(self):
        """Affiche le menu principal"""
        print("\n" + "="*60)
        print(" SYSTÈME DE DIAGNOSTIC CLINIQUE - CANCER DU SEIN")
        print("="*60)
        print("\n1. Nouveau diagnostic")
        print("2. Diagnostic depuis un fichier")
        print("3. Afficher les informations sur les features")
        print("4. Quitter")
        print("-"*60)
    
    def get_patient_data_manual(self):
        """Collecte les données du patient manuellement"""
        print("\n" + "="*60)
        print(" SAISIE DES DONNÉES DU PATIENT")
        print("="*60)
        print("\nEntrez les valeurs pour chaque caractéristique:")
        print("(Vous pouvez entrer 'r' pour générer des valeurs aléatoires)\n")
        
        patient_data = []
        
        for i, feature in enumerate(self.feature_names):
            while True:
                try:
                    value = input(f"{i+1}. {feature}: ")
                    
                    if value.lower() == 'r':
                        # Générer des valeurs aléatoires réalistes
                        patient_data = self._generate_random_patient()
                        print("\nDonnées aléatoires générées!")
                        return np.array(patient_data)
                    
                    patient_data.append(float(value))
                    break
                except ValueError:
                    print("   Erreur: Veuillez entrer un nombre valide ou 'r'")
        
        return np.array(patient_data)
    
    def _generate_random_patient(self):
        """Génère des données de patient aléatoires réalistes"""
        # Valeurs approximatives basées sur le dataset
        return [
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
        ]
    
    def display_result(self, result):
        """Affiche le résultat du diagnostic"""
        print("\n" + "="*60)
        print(" RÉSULTAT DU DIAGNOSTIC")
        print("="*60)
        print(f"\nDiagnostic: {result['diagnosis'].upper()}")
        print(f"\nProbabilité de tumeur maligne: {result['probability_malignant']:.2%}")
        print(f"Probabilité de tumeur bénigne: {result['probability_benign']:.2%}")
        
        if result['diagnosis'] == "infecté":
            print("\n⚠️  ATTENTION: Résultat positif pour tumeur maligne")
            print("   Recommandation: Consultation médicale immédiate requise")
        else:
            print("\n✓  Résultat: Tumeur bénigne détectée")
            print("   Recommandation: Suivi médical régulier conseillé")
        
        print("="*60)
    
    def display_features_info(self):
        """Affiche des informations sur les features"""
        print("\n" + "="*60)
        print(" INFORMATIONS SUR LES CARACTÉRISTIQUES")
        print("="*60)
        print("\nLe modèle utilise 30 caractéristiques:")
        print("\n1. MESURES MOYENNES (mean):")
        print("   - Radius, texture, perimeter, area")
        print("   - Smoothness, compactness, concavity")
        print("   - Concave points, symmetry, fractal dimension")
        print("\n2. ERREURS STANDARD (error):")
        print("   - Mêmes mesures que ci-dessus")
        print("\n3. PIRES VALEURS (worst):")
        print("   - Mêmes mesures que ci-dessus")
        print("\nTotal: 30 caractéristiques numériques")
        print("="*60)
    
    def run(self):
        """Lance l'interface"""
        while True:
            self.display_menu()
            
            choice = input("\nVotre choix: ").strip()
            
            if choice == '1':
                patient_data = self.get_patient_data_manual()
                result = self.api.predict(patient_data)
                self.display_result(result)
                
                input("\nAppuyez sur Entrée pour continuer...")
                
            elif choice == '2':
                print("\nFonctionnalité à venir...")
                input("Appuyez sur Entrée pour continuer...")
                
            elif choice == '3':
                self.display_features_info()
                input("\nAppuyez sur Entrée pour continuer...")
                
            elif choice == '4':
                print("\nMerci d'avoir utilisé le système de diagnostic.")
                print("Au revoir!\n")
                break
                
            else:
                print("\n❌ Choix invalide. Veuillez réessayer.")
                input("Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    interface = ClinicalInterface("../cancer_model.pkl")
    interface.run()