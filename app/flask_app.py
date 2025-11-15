"""Application Flask pour l'interface web du système de prédiction de cancer"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from pipline.trainer import Trainer
from pipline.evaluator import Evaluator
from core.logisticregression import LogisticRegressionModel
from core.neural_network import NeuralNetwork
from core.dataset import Dataset
from utils.metrics import MetricsCalculator
import numpy as np
import pickle
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'cancer_prediction_secret_key_2024'

# Stockage en session des modèles et données
models_storage = {}

AVAILABLE_MODELS = {
    'logistic_regression': {
        'name': 'Régression Logistique',
        'class': LogisticRegressionModel,
        'default_params': {'max_iter': 5000},
        'description': 'Modèle linéaire rapide et interprétable'
    },
    'neural_network': {
        'name': 'Réseau de Neurones',
        'class': NeuralNetwork,
        'default_params': {'hidden_size': 16, 'epochs': 1000, 'learning_rate': 0.01},
        'description': 'Modèle non-linéaire capable de capturer des relations complexes'
    }
}


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html', models=AVAILABLE_MODELS)


@app.route('/select_model')
def select_model():
    """Page de sélection de modèle"""
    return render_template('select_model.html', models=AVAILABLE_MODELS)


@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    """Page d'entraînement de modèle"""
    if request.method == 'GET':
        return render_template('train_model.html', models=AVAILABLE_MODELS)
    
    # POST: Entraîner le modèle
    try:
        model_type = request.form.get('model_type')
        
        if model_type not in AVAILABLE_MODELS:
            return jsonify({'error': 'Type de modèle invalide'}), 400
        
        model_info = AVAILABLE_MODELS[model_type]
        
        # Récupérer les paramètres personnalisés
        custom_params = {}
        if model_type == 'logistic_regression':
            max_iter = request.form.get('max_iter')
            if max_iter:
                custom_params['max_iter'] = int(max_iter)
        elif model_type == 'neural_network':
            hidden_size = request.form.get('hidden_size')
            epochs = request.form.get('epochs')
            learning_rate = request.form.get('learning_rate')
            
            if hidden_size:
                custom_params['hidden_size'] = int(hidden_size)
            if epochs:
                custom_params['epochs'] = int(epochs)
            if learning_rate:
                custom_params['learning_rate'] = float(learning_rate)
        
        # Fusionner avec les paramètres par défaut
        model_params = {**model_info['default_params'], **custom_params}
        
        # Entraîner le modèle
        trainer = Trainer(
            model_class=model_info['class'],
            model_params=model_params
        )
        
        trained_model, X_test, y_test, losses = trainer.run_training(
            test_size=0.2,
            random_state=42,
            calculate_loss=True,
            verbose=False
        )
        
        # Calculer les métriques
        evaluator = Evaluator(trained_model.model if hasattr(trained_model, 'model') else trained_model)
        metrics = evaluator.calculate_metrics(X_test, y_test)
        
        # Stocker le modèle dans la session
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        models_storage[session_id] = {
            'model': trained_model,
            'model_type': model_type,
            'model_name': model_info['name'],
            'X_test': X_test,
            'y_test': y_test,
            'metrics': metrics,
            'losses': losses,
            'params': model_params
        }
        
        session['current_model_id'] = session_id
        
        return jsonify({
            'success': True,
            'model_id': session_id,
            'model_name': model_info['name'],
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score'])
            },
            'losses': {
                'train_log_loss': float(losses['train'].get('log_loss', 0)),
                'test_log_loss': float(losses['test'].get('log_loss', 0))
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Page de prédiction"""
    if request.method == 'GET':
        model_id = session.get('current_model_id')
        model_loaded = model_id in models_storage if model_id else False
        model_name = models_storage[model_id]['model_name'] if model_loaded else None
        
        return render_template('predict.html', 
                             model_loaded=model_loaded,
                             model_name=model_name)
    
    # POST: Faire une prédiction
    try:
        model_id = session.get('current_model_id')
        
        if not model_id or model_id not in models_storage:
            return jsonify({'error': 'Aucun modèle chargé. Veuillez d\'abord entraîner un modèle.'}), 400
        
        model_data = models_storage[model_id]
        model = model_data['model']
        
        # Récupérer les données du patient
        data_source = request.form.get('data_source')
        
        if data_source == 'random':
            # Générer des données aléatoires
            patient_data = _generate_random_patient()
            
        elif data_source == 'test_sample':
            # Utiliser un échantillon de test
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            idx = np.random.randint(0, len(X_test))
            patient_data = X_test[idx]
            true_label = int(y_test[idx])
            
        elif data_source == 'manual':
            # Données manuelles (JSON)
            try:
                features_json = request.form.get('features')
                patient_data = np.array(json.loads(features_json))
            except:
                return jsonify({'error': 'Format de données invalide'}), 400
        else:
            return jsonify({'error': 'Source de données invalide'}), 400
        
        # Faire la prédiction
        patient_data_reshaped = patient_data.reshape(1, -1)
        prediction = model.predict(patient_data_reshaped)[0]
        
        # Obtenir les probabilités
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(patient_data_reshaped)[0]
            if len(proba) > 1:
                prob_benign = float(proba[0])
                prob_malignant = float(proba[1])
            else:
                prob_malignant = float(proba[0])
                prob_benign = 1 - prob_malignant
        else:
            prob_malignant = float(prediction)
            prob_benign = 1 - prob_malignant
        
        result = {
            'success': True,
            'model_name': model_data['model_name'],
            'prediction': 'Malin' if prediction == 1 else 'Bénin',
            'prediction_value': int(prediction),
            'prob_benign': prob_benign,
            'prob_malignant': prob_malignant,
            'data_source': data_source
        }
        
        # Ajouter la vraie valeur si échantillon de test
        if data_source == 'test_sample':
            result['true_label'] = 'Malin' if true_label == 1 else 'Bénin'
            result['correct'] = (prediction == true_label)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate')
def evaluate():
    """Page d'évaluation du modèle"""
    model_id = session.get('current_model_id')
    
    if not model_id or model_id not in models_storage:
        return render_template('evaluate.html', model_loaded=False)
    
    model_data = models_storage[model_id]
    
    return render_template('evaluate.html',
                         model_loaded=True,
                         model_name=model_data['model_name'],
                         metrics=model_data['metrics'],
                         losses=model_data['losses'])


@app.route('/compare_models', methods=['GET', 'POST'])
def compare_models():
    """Page de comparaison des modèles"""
    if request.method == 'GET':
        return render_template('compare.html')
    
    # POST: Comparer tous les modèles
    try:
        results = {}
        
        for model_key, model_info in AVAILABLE_MODELS.items():
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
            
            evaluator = Evaluator(model.model if hasattr(model, 'model') else model)
            metrics = evaluator.calculate_metrics(X_test, y_test)
            
            results[model_key] = {
                'name': model_info['name'],
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'log_loss': float(losses['test'].get('log_loss', 0))
            }
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info/<model_type>')
def model_info(model_type):
    """Retourne les informations sur un modèle"""
    if model_type not in AVAILABLE_MODELS:
        return jsonify({'error': 'Modèle non trouvé'}), 404
    
    model_info = AVAILABLE_MODELS[model_type]
    return jsonify({
        'name': model_info['name'],
        'description': model_info['description'],
        'default_params': model_info['default_params']
    })


@app.route('/save_model', methods=['POST'])
def save_model():
    """Sauvegarder le modèle actuel"""
    try:
        model_id = session.get('current_model_id')
        
        if not model_id or model_id not in models_storage:
            return jsonify({'error': 'Aucun modèle à sauvegarder'}), 400
        
        filename = request.form.get('filename', 'model.pkl')
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        model = models_storage[model_id]['model']
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        return jsonify({'success': True, 'filename': filename})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _generate_random_patient():
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
