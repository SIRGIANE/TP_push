"""Script principal pour entraîner et sauvegarder le modèle"""
from pipline.trainer import Trainer
from pipline.evaluator import Evaluator
from utils.metrics import MetricsCalculator
from core.dataset import Dataset

def main():
    print("="*60)
    print(" ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DE CANCER")
    print("="*60)
    
    # 1. Initialiser le trainer
    print("\n1. Initialisation du trainer...")
    trainer = Trainer()
    
    # 2. Entraîner le modèle avec calcul automatique des pertes
    print("2. Entraînement du modèle avec calcul des pertes...")
    model, X_test, y_test, losses = trainer.run_training(
        test_size=0.2,
        random_state=42,
        calculate_loss=True,
        verbose=True,
        max_iter=5000
    )
    
    # 3. Évaluer le modèle (métriques classiques)
    print("\n3. Évaluation avec métriques de classification...")
    evaluator = Evaluator(model.model)
    metrics = evaluator.print_metrics(X_test, y_test)
    
    # 4. Afficher des métriques supplémentaires
    print("\n4. Métriques détaillées...")
    y_pred = model.predict(X_test)
    
    dataset = Dataset()
    dataset.load_data()
    target_names = dataset.get_target_names()
    
    metrics_calc = MetricsCalculator()
    metrics_calc.print_confusion_matrix(y_test, y_pred, target_names)
    metrics_calc.print_classification_report(y_test, y_pred, target_names)
    
    # 5. Afficher le résumé de l'entraînement
    print("\n5. Résumé de l'entraînement...")
    trainer.print_summary()
    
    # 6. Sauvegarder le modèle
    print("\n6. Sauvegarde du modèle...")
    trainer.save_model("cancer_model.pkl")
    
    print("\n" + "="*60)
    print(" ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print("\nVous pouvez maintenant utiliser:")
    print("  - python app/api.py pour tester l'API")
    print("  - python app/interfaceclinique.py pour l'interface CLI")
    print("  - python test_loss.py pour tester le module loss")
    print("="*60)


def main_with_detailed_report():
    """Version avec rapport détaillé"""
    trainer = Trainer()
    model, X_test, y_test = trainer.run_training_with_report(
        test_size=0.2,
        random_state=42,
        max_iter=5000
    )
    
    # Évaluation finale
    print("\n" + "="*60)
    print(" ÉVALUATION FINALE")
    print("="*60)
    
    evaluator = Evaluator(model.model)
    evaluator.print_metrics(X_test, y_test)
    
    # Sauvegarder
    trainer.save_model("cancer_model.pkl")
    
    print("\n✓ Processus terminé avec succès!")


if __name__ == "__main__":
    import sys
    
    # Choisir le mode d'exécution
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        main_with_detailed_report()
    else:
        main()