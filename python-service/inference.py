#!/usr/bin/env python3
"""
Production Career Prediction Inference Service
==============================================
High-performance inference wrapper with explainability and caching.

Author: Senior ML Engineer
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available, using feature importance fallback")

class CareerPredictor:
    """Production-ready career prediction inference service."""
    
    def __init__(self, model_path: str = None, encoder_path: str = None, metadata_path: str = None):
        """Initialize the career predictor with model artifacts."""
        
        # Default paths
        self.model_path = model_path or 'models/career_field_model_latest.pkl'
        self.encoder_path = encoder_path or 'models/label_encoder_latest.pkl'
        self.metadata_path = metadata_path or 'models/model_metadata_latest.json'
        
        # Fallback paths for backward compatibility
        if not os.path.exists(self.model_path):
            self.model_path = 'career_field_model.pkl'
        if not os.path.exists(self.encoder_path):
            self.encoder_path = 'label_encoder.pkl'
        
        self.model = None
        self.label_encoder = None
        self.metadata = None
        self.explainer = None
        
        self._load_model_artifacts()
        self._initialize_explainer()
    
    def _load_model_artifacts(self):
        """Load model, encoder, and metadata."""
        
        try:
            print(f"📦 Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            print(f"📦 Loading label encoder from {self.encoder_path}")
            self.label_encoder = joblib.load(self.encoder_path)
            
            if os.path.exists(self.metadata_path):
                print(f"📦 Loading metadata from {self.metadata_path}")
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                print("⚠️  Metadata file not found, using defaults")
                self.metadata = {
                    'model_info': {'version': 'unknown'},
                    'performance_metrics': {'cv_accuracy': 0.0}
                }
            
            print("✅ Model artifacts loaded successfully")
            print(f"   Model version: {self.metadata['model_info'].get('version', 'unknown')}")
            print(f"   CV accuracy: {self.metadata['performance_metrics'].get('cv_accuracy', 0.0):.3f}")
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            raise
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer for model interpretability."""
        
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available, explainability will use feature importance")
            return
        
        try:
            # Create a small sample for explainer initialization
            sample_data = self._create_sample_data()
            
            # Initialize TreeExplainer for tree-based models
            if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                base_model = self.model.named_steps['model']
                if hasattr(base_model, 'estimators_'):  # Stacking ensemble
                    # Use the first tree-based estimator for explanation
                    for name, estimator in base_model.estimators_:
                        if 'rf' in name.lower() or 'xgb' in name.lower():
                            try:
                                self.explainer = shap.TreeExplainer(estimator)
                                print("✅ SHAP TreeExplainer initialized")
                                break
                            except:
                                continue
            
            if self.explainer is None:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    sample_data[:50]  # Use small sample for efficiency
                )
                print("✅ SHAP KernelExplainer initialized")
                
        except Exception as e:
            print(f"⚠️  Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _create_sample_data(self) -> np.ndarray:
        """Create sample data for explainer initialization."""
        
        # Expected feature order
        feature_names = [
            'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
            'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
            'response_consistency', 'response_time_variance', 'straight_lining_score',
            'completion_rate', 'average_response_time', 'duration_seconds'
        ]
        
        # Generate sample data with realistic ranges
        np.random.seed(42)
        n_samples = 100
        
        sample_data = []
        for _ in range(n_samples):
            sample = {
                # Big Five (1-10 scale)
                'Openness': np.random.uniform(1, 10),
                'Conscientiousness': np.random.uniform(1, 10),
                'Extraversion': np.random.uniform(1, 10),
                'Agreeableness': np.random.uniform(1, 10),
                'Neuroticism': np.random.uniform(1, 10),
                
                # RIASEC (1-10 scale)
                'Realistic': np.random.uniform(1, 10),
                'Investigative': np.random.uniform(1, 10),
                'Artistic': np.random.uniform(1, 10),
                'Social': np.random.uniform(1, 10),
                'Enterprising': np.random.uniform(1, 10),
                'Conventional': np.random.uniform(1, 10),
                
                # Behavioral features
                'response_consistency': np.random.uniform(0.5, 2.5),
                'response_time_variance': np.random.uniform(500, 3000),
                'straight_lining_score': np.random.uniform(0.0, 0.3),
                'completion_rate': np.random.uniform(0.95, 1.0),
                'average_response_time': np.random.uniform(1500, 8000),
                'duration_seconds': np.random.uniform(300, 1200)
            }
            sample_data.append([sample[col] for col in feature_names])
        
        return np.array(sample_data)
    
    def predict_top3(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict top 3 career matches with explanations.
        
        Args:
            input_data: Dictionary containing trait scores and behavioral features
            
        Returns:
            Dictionary with predictions, probabilities, and explanations
        """
        
        try:
            # Prepare input features
            features = self._prepare_features(input_data)
            
            # Make predictions
            probabilities = self.model.predict_proba(features)[0]
            
            # Get top 3 predictions
            top3_indices = np.argsort(probabilities)[::-1][:3]
            top3_careers = self.label_encoder.inverse_transform(top3_indices)
            top3_probabilities = probabilities[top3_indices]
            
            # Format predictions
            predictions = []
            for career, prob in zip(top3_careers, top3_probabilities):
                predictions.append({
                    'career': career,
                    'probability': float(prob)
                })
            
            # Generate explanations
            explanations = self._generate_explanations(features, top3_indices)
            
            # Extract trait scores for response
            trait_scores = self._extract_trait_scores(input_data)
            
            result = {
                'predictions': predictions,
                'trait_scores': trait_scores,
                'explanations': explanations,
                'model_version': self.metadata['model_info'].get('version', 'unknown'),
                'confidence_score': float(top3_probabilities[0])  # Top prediction confidence
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'error': str(e),
                'predictions': [],
                'trait_scores': {},
                'explanations': {},
                'model_version': 'error'
            }
    
    def _prepare_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Prepare input features for model prediction."""
        
        # Expected feature order
        feature_names = [
            'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
            'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
            'response_consistency', 'response_time_variance', 'straight_lining_score',
            'completion_rate', 'average_response_time', 'duration_seconds'
        ]
        
        # Handle different input formats
        if 'trait_scores' in input_data:
            # Input contains pre-calculated trait scores
            trait_data = input_data['trait_scores']
            
            # Extract Big Five
            big_five = trait_data.get('big_five', {})
            riasec = trait_data.get('riasec', {})
            
            features = {
                'Openness': big_five.get('Openness', 5.0),
                'Conscientiousness': big_five.get('Conscientiousness', 5.0),
                'Extraversion': big_five.get('Extraversion', 5.0),
                'Agreeableness': big_five.get('Agreeableness', 5.0),
                'Neuroticism': big_five.get('Neuroticism', 5.0),
                'Realistic': riasec.get('Realistic', 5.0),
                'Investigative': riasec.get('Investigative', 5.0),
                'Artistic': riasec.get('Artistic', 5.0),
                'Social': riasec.get('Social', 5.0),
                'Enterprising': riasec.get('Enterprising', 5.0),
                'Conventional': riasec.get('Conventional', 5.0)
            }
        else:
            # Direct feature input
            features = {
                'Openness': input_data.get('Openness', 5.0),
                'Conscientiousness': input_data.get('Conscientiousness', 5.0),
                'Extraversion': input_data.get('Extraversion', 5.0),
                'Agreeableness': input_data.get('Agreeableness', 5.0),
                'Neuroticism': input_data.get('Neuroticism', 5.0),
                'Realistic': input_data.get('Realistic', 5.0),
                'Investigative': input_data.get('Investigative', 5.0),
                'Artistic': input_data.get('Artistic', 5.0),
                'Social': input_data.get('Social', 5.0),
                'Enterprising': input_data.get('Enterprising', 5.0),
                'Conventional': input_data.get('Conventional', 5.0)
            }
        
        # Add behavioral features with defaults
        derived_features = input_data.get('derived_features', {})
        features.update({
            'response_consistency': derived_features.get('response_consistency', 1.5),
            'response_time_variance': derived_features.get('response_time_variance', 2000),
            'straight_lining_score': derived_features.get('straight_lining_score', 0.1),
            'completion_rate': derived_features.get('completion_rate', 1.0),
            'average_response_time': derived_features.get('average_response_time', 4000),
            'duration_seconds': derived_features.get('duration_seconds', 600)
        })
        
        # Convert to array in correct order
        feature_array = np.array([[features[col] for col in feature_names]])
        
        return feature_array
    
    def _extract_trait_scores(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trait scores for response."""
        
        if 'trait_scores' in input_data:
            return input_data['trait_scores']
        
        # Build trait scores from direct input
        big_five = {
            'Openness': input_data.get('Openness', 5.0),
            'Conscientiousness': input_data.get('Conscientiousness', 5.0),
            'Extraversion': input_data.get('Extraversion', 5.0),
            'Agreeableness': input_data.get('Agreeableness', 5.0),
            'Neuroticism': input_data.get('Neuroticism', 5.0)
        }
        
        riasec = {
            'Realistic': input_data.get('Realistic', 5.0),
            'Investigative': input_data.get('Investigative', 5.0),
            'Artistic': input_data.get('Artistic', 5.0),
            'Social': input_data.get('Social', 5.0),
            'Enterprising': input_data.get('Enterprising', 5.0),
            'Conventional': input_data.get('Conventional', 5.0)
        }
        
        return {'big_five': big_five, 'riasec': riasec}
    
    def _generate_explanations(self, features: np.ndarray, top_indices: np.ndarray) -> Dict[str, Any]:
        """Generate explanations for predictions."""
        
        explanations = {
            'method': 'feature_importance',
            'top_features': [],
            'feature_contributions': {}
        }
        
        try:
            if SHAP_AVAILABLE and self.explainer is not None:
                # Use SHAP for explanations
                shap_values = self.explainer.shap_values(features)
                
                if isinstance(shap_values, list):
                    # Multi-class case - use values for top prediction
                    shap_vals = shap_values[top_indices[0]]
                else:
                    shap_vals = shap_values
                
                # Get feature names
                feature_names = [
                    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
                    'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
                    'response_consistency', 'response_time_variance', 'straight_lining_score',
                    'completion_rate', 'average_response_time', 'duration_seconds'
                ]
                
                # Get top contributing features
                if len(shap_vals.shape) > 1:
                    feature_importance = np.abs(shap_vals[0])
                else:
                    feature_importance = np.abs(shap_vals)
                
                top_feature_indices = np.argsort(feature_importance)[::-1][:5]
                
                explanations['method'] = 'shap'
                explanations['top_features'] = [
                    [feature_names[i], float(feature_importance[i])]
                    for i in top_feature_indices
                ]
                
            else:
                # Fallback to model feature importance
                if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                    base_model = self.model.named_steps['model']
                    
                    # Try to get feature importance from base models
                    if hasattr(base_model, 'estimators_'):
                        for name, estimator in base_model.estimators_:
                            if hasattr(estimator, 'feature_importances_'):
                                importance = estimator.feature_importances_
                                feature_names = [
                                    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
                                    'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
                                    'response_consistency', 'response_time_variance', 'straight_lining_score',
                                    'completion_rate', 'average_response_time', 'duration_seconds'
                                ]
                                
                                # Adjust for engineered features
                                if len(importance) > len(feature_names):
                                    feature_names.extend([f'engineered_feature_{i}' for i in range(len(importance) - len(feature_names))])
                                
                                top_indices_feat = np.argsort(importance)[::-1][:5]
                                explanations['top_features'] = [
                                    [feature_names[i] if i < len(feature_names) else f'feature_{i}', float(importance[i])]
                                    for i in top_indices_feat
                                ]
                                break
                
        except Exception as e:
            print(f"⚠️  Error generating explanations: {e}")
            explanations['error'] = str(e)
        
        return explanations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        
        return {
            'model_version': self.metadata['model_info'].get('version', 'unknown'),
            'training_date': self.metadata['model_info'].get('training_date', 'unknown'),
            'model_type': self.metadata['model_info'].get('model_type', 'unknown'),
            'cv_accuracy': self.metadata['performance_metrics'].get('cv_accuracy', 0.0),
            'test_accuracy': self.metadata['performance_metrics'].get('test_accuracy', 0.0),
            'top3_accuracy': self.metadata['performance_metrics'].get('top3_accuracy', 0.0),
            'n_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist()
        }

# Global predictor instance for Flask app
_predictor = None

def get_predictor() -> CareerPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CareerPredictor()
    return _predictor

def predict_career(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for career prediction."""
    predictor = get_predictor()
    return predictor.predict_top3(input_data)

def get_model_info() -> Dict[str, Any]:
    """Convenience function to get model information."""
    predictor = get_predictor()
    return predictor.get_model_info()

# Example usage
if __name__ == "__main__":
    # Test the predictor
    predictor = CareerPredictor()
    
    # Sample input
    sample_input = {
        'trait_scores': {
            'big_five': {
                'Openness': 8.5,
                'Conscientiousness': 7.2,
                'Extraversion': 6.1,
                'Agreeableness': 7.8,
                'Neuroticism': 3.4
            },
            'riasec': {
                'Realistic': 4.2,
                'Investigative': 8.7,
                'Artistic': 6.3,
                'Social': 7.1,
                'Enterprising': 5.9,
                'Conventional': 6.8
            }
        },
        'derived_features': {
            'response_consistency': 1.2,
            'response_time_variance': 1800,
            'straight_lining_score': 0.05,
            'completion_rate': 1.0,
            'average_response_time': 3500,
            'duration_seconds': 480
        }
    }
    
    # Make prediction
    result = predictor.predict_top3(sample_input)
    
    print("🧪 Test Prediction Results:")
    print(json.dumps(result, indent=2))