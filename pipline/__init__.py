"""Package pipline pour l'entraînement et l'évaluation"""
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = ['Trainer', 'Evaluator']