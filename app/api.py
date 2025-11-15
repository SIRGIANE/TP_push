"""Module API pour l'utilisation du modèle"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import ClinicalPredictor
from core.logisticregression import LogisticRegressionModel
import numpy as np


class CancerPredictionAPI:
    """API pour effectuer des prédictions de cancer"""
    
    def __init__(self, model_path="cancer_model.pkl"):
        """
        Initialise l'API avec un modèle sauvegardé
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.model = LogisticRegressionModel()
        self.model.load(model_path)
        self.predictor = ClinicalPredictor(self.model.model)
        
    def predict(self, patient_data):
        """
        Fait une prédiction pour un patient
        
        Args:
            patient_data: Array ou liste contenant les features du patient
            
        Returns:
            dict: Résultat du diagnostic
        """
        if isinstance(patient_data, list):
            patient_data = np.array(patient_data).reshape(1, -1)
        elif len(patient_data.shape) == 1:
            patient_data = patient_data.reshape(1, -1)
            
        diagnosis = self.predictor.diagnose(patient_data)
        probability = self.predictor.get_probability(patient_data)
        
        return {
            'diagnosis': diagnosis,
            'probability_malignant': float(probability),
            'probability_benign': float(1 - probability)
        }
    
    def predict_batch(self, patients_data):
        """
        Fait des prédictions pour plusieurs patients
        
        Args:
            patients_data: Array contenant les features de plusieurs patients
            
        Returns:
            list: Liste de résultats
        """
        diagnoses = self.predictor.diagnose_batch(patients_data)
        probs = self.predictor.model.predict_proba(patients_data)[:, 1]
        
        results = []
        for i, (diag, prob) in enumerate(zip(diagnoses, probs)):
            results.append({
                'patient_id': i,
                'diagnosis': diag,
                'probability_malignant': float(prob),
                'probability_benign': float(1 - prob)
            })
        
        return results


if __name__ == "__main__":
    # Exemple d'utilisation
    api = CancerPredictionAPI("../cancer_model.pkl")
    
    # Exemple de données patient
    test_patient = np.random.rand(30)  # 30 features
    
    result = api.predict(test_patient)
    print("Résultat de la prédiction:")
    print(f"Diagnostic: {result['diagnosis']}")
    print(f"Probabilité malignant: {result['probability_malignant']:.2%}")
    print(f"Probabilité benign: {result['probability_benign']:.2%}")