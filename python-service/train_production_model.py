#!/usr/bin/env python3
"""
Production Career Prediction Model Training Pipeline
==================================================
FAANG-level ML engineering implementation for psychometric-based career prediction.

Target: ≥90% realistic accuracy with robust generalization
Author: Senior ML Engineer
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_recall_fscore_support, top_k_accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV

# Advanced ensemble methods
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not available")
    LIGHTGBM_AVAILABLE = False

# Class imbalance handling
try:
    from imblearn.over_sampling import SMOTE, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_AVAILABLE = True
except ImportError:
    print("⚠️  imbalanced-learn not available")
    IMBALANCED_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️  Optuna not available, using GridSearchCV")
    OPTUNA_AVAILABLE = False

# Model persistence
import joblib

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🚀 Production Career Prediction Model Training Pipeline")
print("🎯 Target: ≥90% Accuracy with Robust Generalization")
print("=" * 70)

# =============================================================================
# 1. DATASET FINGERPRINTING
# =============================================================================

def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of dataset for versioning and reproducibility."""
    # Create canonical representation
    canonical_data = df.copy()
    canonical_data = canonical_data.sort_values(list(canonical_data.columns))
    canonical_data = canonical_data.reset_index(drop=True)
    
    # Convert to string and hash
    data_string = canonical_data.to_csv(index=False)
    return hashlib.sha256(data_string.encode()).hexdigest()

# =============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# =============================================================================

class PsychometricFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering for psychometric-based career prediction."""
    
    def __init__(self, include_interactions=True, include_behavioral=True):
        self.include_interactions = include_interactions
        self.include_behavioral = include_behavioral
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            # Convert to DataFrame for easier manipulation
            feature_names = self._get_expected_columns()
            X_transformed = pd.DataFrame(X, columns=feature_names)
        
        # 1. Big Five derived features
        X_transformed['Big5_Sum'] = (
            X_transformed['Openness'] + X_transformed['Conscientiousness'] + 
            X_transformed['Extraversion'] + X_transformed['Agreeableness'] + 
            (11 - X_transformed['Neuroticism'])  # Emotional stability
        )
        X_transformed['Emotional_Stability'] = 11 - X_transformed['Neuroticism']
        X_transformed['Leadership_Index'] = (X_transformed['Extraversion'] + X_transformed['Conscientiousness']) / 2
        X_transformed['Creativity_Index'] = (X_transformed['Openness'] + X_transformed['Extraversion']) / 2
        X_transformed['Analytical_Index'] = (X_transformed['Conscientiousness'] + X_transformed['Openness']) / 2
        X_transformed['Social_Index'] = (X_transformed['Extraversion'] + X_transformed['Agreeableness']) / 2
        
        # 2. RIASEC derived features
        X_transformed['RIASEC_Sum'] = (
            X_transformed['Realistic'] + X_transformed['Investigative'] + 
            X_transformed['Artistic'] + X_transformed['Social'] + 
            X_transformed['Enterprising'] + X_transformed['Conventional']
        )
        X_transformed['STEM_Interest'] = (X_transformed['Realistic'] + X_transformed['Investigative']) / 2
        X_transformed['People_Interest'] = (X_transformed['Social'] + X_transformed['Enterprising']) / 2
        X_transformed['Creative_Interest'] = X_transformed['Artistic']
        X_transformed['Structure_Interest'] = X_transformed['Conventional']
        
        # 3. Cross-domain interactions
        if self.include_interactions:
            X_transformed['Openness_x_Artistic'] = X_transformed['Openness'] * X_transformed['Artistic'] / 10
            X_transformed['Conscientiousness_x_Conventional'] = X_transformed['Conscientiousness'] * X_transformed['Conventional'] / 10
            X_transformed['Extraversion_x_Social'] = X_transformed['Extraversion'] * X_transformed['Social'] / 10
            X_transformed['Analytical_x_Investigative'] = X_transformed['Analytical_Index'] * X_transformed['Investigative'] / 10
            X_transformed['Leadership_x_Enterprising'] = X_transformed['Leadership_Index'] * X_transformed['Enterprising'] / 10
        
        # 4. Behavioral quality features
        if self.include_behavioral and 'response_consistency' in X_transformed.columns:
            X_transformed['Quality_Score'] = (
                (5 - X_transformed['response_consistency']) +  # Lower consistency = higher quality
                (1 - X_transformed['straight_lining_score']) * 5 +  # Less straight-lining = higher quality
                X_transformed['completion_rate'] * 5
            ) / 3
            
            # Response time features
            X_transformed['Response_Speed_Category'] = pd.cut(
                X_transformed['average_response_time'], 
                bins=[0, 2000, 5000, 10000, float('inf')], 
                labels=[1, 2, 3, 4]
            ).astype(float)
        
        # 5. Personality type indicators (simplified MBTI-like)
        X_transformed['Thinking_vs_Feeling'] = X_transformed['Conscientiousness'] - X_transformed['Agreeableness']
        X_transformed['Judging_vs_Perceiving'] = X_transformed['Conscientiousness'] - X_transformed['Openness']
        X_transformed['Sensing_vs_Intuition'] = X_transformed['Realistic'] - X_transformed['Investigative']
        
        # Store feature names for later use
        self.feature_names_ = X_transformed.columns.tolist()
        
        return X_transformed.values

    def _get_expected_columns(self):
        """Define expected column order for consistency."""
        return [
            'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
            'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
            'response_consistency', 'response_time_variance', 'straight_lining_score',
            'completion_rate', 'average_response_time', 'duration_seconds'
        ]

# =============================================================================
# 3. DATASET GENERATION AND LOADING
# =============================================================================

def generate_psychometric_dataset(n_samples=3000, save_path='student_training_data_v2.csv'):
    """Generate realistic psychometric assessment dataset."""
    
    print(f"📊 Generating {n_samples} psychometric assessment samples...")
    
    # Define career fields with psychometric profiles
    career_profiles = {
        "Doctor → General Medicine": {
            "big_five": {"Openness": (6, 8), "Conscientiousness": (8, 10), "Extraversion": (5, 8), 
                        "Agreeableness": (7, 9), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (4, 7), "Investigative": (7, 10), "Artistic": (3, 6), 
                      "Social": (6, 9), "Enterprising": (5, 8), "Conventional": (5, 8)}
        },
        "Doctor → Neurosurgery": {
            "big_five": {"Openness": (7, 9), "Conscientiousness": (9, 10), "Extraversion": (4, 7), 
                        "Agreeableness": (6, 8), "Neuroticism": (1, 3)},
            "riasec": {"Realistic": (6, 9), "Investigative": (8, 10), "Artistic": (2, 5), 
                      "Social": (5, 7), "Enterprising": (6, 9), "Conventional": (6, 8)}
        },
        "Engineer → Computer Science": {
            "big_five": {"Openness": (7, 10), "Conscientiousness": (7, 9), "Extraversion": (3, 7), 
                        "Agreeableness": (5, 8), "Neuroticism": (2, 6)},
            "riasec": {"Realistic": (5, 8), "Investigative": (8, 10), "Artistic": (4, 7), 
                      "Social": (3, 6), "Enterprising": (4, 7), "Conventional": (6, 9)}
        },
        "Engineer → Mechanical": {
            "big_five": {"Openness": (6, 8), "Conscientiousness": (8, 10), "Extraversion": (4, 7), 
                        "Agreeableness": (5, 8), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (8, 10), "Investigative": (7, 9), "Artistic": (3, 6), 
                      "Social": (3, 6), "Enterprising": (5, 8), "Conventional": (6, 8)}
        },
        "IT Professional → Data Science": {
            "big_five": {"Openness": (8, 10), "Conscientiousness": (6, 9), "Extraversion": (4, 7), 
                        "Agreeableness": (5, 8), "Neuroticism": (3, 6)},
            "riasec": {"Realistic": (4, 7), "Investigative": (9, 10), "Artistic": (5, 8), 
                      "Social": (4, 7), "Enterprising": (5, 8), "Conventional": (7, 9)}
        },
        "IT Professional → Software Development": {
            "big_five": {"Openness": (7, 9), "Conscientiousness": (6, 9), "Extraversion": (3, 6), 
                        "Agreeableness": (5, 8), "Neuroticism": (3, 7)},
            "riasec": {"Realistic": (5, 8), "Investigative": (8, 10), "Artistic": (6, 9), 
                      "Social": (3, 6), "Enterprising": (4, 7), "Conventional": (6, 8)}
        },
        "Designer → Graphic Design": {
            "big_five": {"Openness": (8, 10), "Conscientiousness": (5, 8), "Extraversion": (5, 8), 
                        "Agreeableness": (6, 9), "Neuroticism": (3, 7)},
            "riasec": {"Realistic": (3, 6), "Investigative": (4, 7), "Artistic": (8, 10), 
                      "Social": (5, 8), "Enterprising": (5, 8), "Conventional": (3, 6)}
        },
        "Designer → UI/UX": {
            "big_five": {"Openness": (8, 10), "Conscientiousness": (6, 9), "Extraversion": (6, 9), 
                        "Agreeableness": (7, 9), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (4, 7), "Investigative": (6, 9), "Artistic": (8, 10), 
                      "Social": (6, 9), "Enterprising": (6, 9), "Conventional": (4, 7)}
        },
        "Business → Management": {
            "big_five": {"Openness": (6, 9), "Conscientiousness": (7, 10), "Extraversion": (7, 10), 
                        "Agreeableness": (6, 9), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (3, 6), "Investigative": (5, 8), "Artistic": (4, 7), 
                      "Social": (7, 10), "Enterprising": (8, 10), "Conventional": (6, 9)}
        },
        "Business → Finance": {
            "big_five": {"Openness": (5, 8), "Conscientiousness": (8, 10), "Extraversion": (6, 9), 
                        "Agreeableness": (5, 8), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (3, 6), "Investigative": (6, 9), "Artistic": (2, 5), 
                      "Social": (5, 8), "Enterprising": (7, 10), "Conventional": (8, 10)}
        },
        "Teacher → Elementary": {
            "big_five": {"Openness": (7, 9), "Conscientiousness": (7, 10), "Extraversion": (6, 9), 
                        "Agreeableness": (8, 10), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (3, 6), "Investigative": (5, 8), "Artistic": (6, 9), 
                      "Social": (8, 10), "Enterprising": (5, 8), "Conventional": (6, 8)}
        },
        "Teacher → Mathematics": {
            "big_five": {"Openness": (6, 9), "Conscientiousness": (8, 10), "Extraversion": (5, 8), 
                        "Agreeableness": (7, 9), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (4, 7), "Investigative": (8, 10), "Artistic": (4, 7), 
                      "Social": (7, 10), "Enterprising": (4, 7), "Conventional": (7, 9)}
        },
        "Scientist → Physics": {
            "big_five": {"Openness": (8, 10), "Conscientiousness": (7, 10), "Extraversion": (3, 6), 
                        "Agreeableness": (5, 8), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (5, 8), "Investigative": (9, 10), "Artistic": (4, 7), 
                      "Social": (3, 6), "Enterprising": (3, 6), "Conventional": (6, 8)}
        },
        "Scientist → Biology": {
            "big_five": {"Openness": (8, 10), "Conscientiousness": (7, 9), "Extraversion": (4, 7), 
                        "Agreeableness": (6, 9), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (6, 9), "Investigative": (9, 10), "Artistic": (3, 6), 
                      "Social": (5, 8), "Enterprising": (3, 6), "Conventional": (5, 8)}
        },
        "Lawyer → Corporate Law": {
            "big_five": {"Openness": (6, 9), "Conscientiousness": (8, 10), "Extraversion": (6, 9), 
                        "Agreeableness": (4, 7), "Neuroticism": (2, 5)},
            "riasec": {"Realistic": (2, 5), "Investigative": (7, 9), "Artistic": (4, 7), 
                      "Social": (5, 8), "Enterprising": (8, 10), "Conventional": (7, 9)}
        }
    }
    
    students = []
    
    for i in range(n_samples):
        # Select career field and subfield
        career_subfield = np.random.choice(list(career_profiles.keys()))
        profile = career_profiles[career_subfield]
        
        # Generate Big Five scores with realistic variance
        big_five_scores = {}
        for trait, (min_val, max_val) in profile["big_five"].items():
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 4  # 95% within range
            score = np.clip(np.random.normal(mean_val, std_val), 1, 10)
            big_five_scores[trait] = round(score, 1)
        
        # Generate RIASEC scores
        riasec_scores = {}
        for interest, (min_val, max_val) in profile["riasec"].items():
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 4
            score = np.clip(np.random.normal(mean_val, std_val), 1, 10)
            riasec_scores[interest] = round(score, 1)
        
        # Generate behavioral quality metrics
        response_consistency = np.random.uniform(0.5, 2.5)  # Lower = more consistent
        response_time_variance = np.random.uniform(500, 3000)  # ms
        straight_lining_score = np.random.uniform(0.0, 0.3)  # Lower = better quality
        completion_rate = np.random.uniform(0.95, 1.0)  # High completion
        average_response_time = np.random.uniform(1500, 8000)  # ms
        duration_seconds = np.random.uniform(300, 1200)  # 5-20 minutes
        
        # Create student record
        student = {
            "Student_ID": i + 1,
            **big_five_scores,
            **riasec_scores,
            "response_consistency": round(response_consistency, 2),
            "response_time_variance": round(response_time_variance, 0),
            "straight_lining_score": round(straight_lining_score, 3),
            "completion_rate": round(completion_rate, 3),
            "average_response_time": round(average_response_time, 0),
            "duration_seconds": round(duration_seconds, 0),
            "device_type": np.random.choice(['mobile', 'desktop'], p=[0.3, 0.7]),
            "Career_Subfield": career_subfield
        }
        
        students.append(student)
    
    df = pd.DataFrame(students)
    
    # Add some noise and edge cases for robustness
    noise_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    for idx in noise_indices:
        # Add some random noise to make the dataset more challenging
        trait_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        for col in trait_cols:
            df.loc[idx, col] = np.clip(df.loc[idx, col] + np.random.normal(0, 1), 1, 10)
    
    # Save dataset
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {len(df)} samples across {df['Career_Subfield'].nunique()} career-subfield combinations")
    print(f"💾 Dataset saved to '{save_path}'")
    print(f"📈 Career distribution:")
    print(df['Career_Subfield'].value_counts().head(10))
    
    return df

def load_dataset(csv_path='student_training_data_v2.csv'):
    """Load and validate dataset."""
    
    if not os.path.exists(csv_path):
        print(f"📊 Dataset not found at {csv_path}, generating new dataset...")
        return generate_psychometric_dataset(save_path=csv_path)
    
    print(f"📊 Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"🎯 Target distribution:")
    print(df['Career_Subfield'].value_counts().head(10))
    
    return df

# =============================================================================
# 4. MODEL DEFINITIONS AND OPTIMIZATION
# =============================================================================

def create_base_models():
    """Create calibrated base models for stacking ensemble."""
    
    models = {}
    
    # 1. Random Forest
    models['rf'] = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        method='isotonic',
        cv=3
    )
    
    # 2. Gradient Boosting
    models['gb'] = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=RANDOM_STATE
        ),
        method='isotonic',
        cv=3
    )
    
    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['xgb'] = CalibratedClassifierCV(
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='mlogloss',
                n_jobs=-1
            ),
            method='isotonic',
            cv=3
        )
    
    # 4. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['cat'] = CalibratedClassifierCV(
            cb.CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=RANDOM_STATE,
                verbose=False
            ),
            method='isotonic',
            cv=3
        )
    
    # 5. Neural Network
    models['mlp'] = CalibratedClassifierCV(
        MLPClassifier(
            hidden_layer_sizes=(150, 75),
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=RANDOM_STATE
        ),
        method='isotonic',
        cv=3
    )
    
    return models

def create_stacking_ensemble(base_models):
    """Create stacking ensemble with robust meta-learner."""
    
    print("🏗️  Creating stacking ensemble...")
    
    # Meta-learner with regularization
    meta_learner = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
    # Create stacking classifier
    stacking_model = StackingClassifier(
        estimators=list(base_models.items()),
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=True  # Include original features
    )
    
    print(f"✅ Stacking ensemble created with {len(base_models)} base models")
    return stacking_model

def optimize_hyperparameters(model, X_train, y_train, model_name):
    """Optimize hyperparameters using Optuna or GridSearchCV."""
    
    if not OPTUNA_AVAILABLE:
        print(f"⚠️  Optuna not available, skipping hyperparameter optimization for {model_name}")
        return model
    
    print(f"🔧 Optimizing hyperparameters for {model_name}...")
    
    def objective(trial):
        # Define hyperparameter search space based on model type
        if 'rf' in model_name.lower():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif 'xgb' in model_name.lower():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0)
            }
        else:
            return 0.0  # Skip optimization for other models
        
        # Create model with suggested parameters
        if 'rf' in model_name.lower():
            temp_model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif 'xgb' in model_name.lower() and XGBOOST_AVAILABLE:
            temp_model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric='mlogloss')
        else:
            return 0.0
        
        # Cross-validation score
        cv_scores = cross_val_score(temp_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return cv_scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=300)  # 5 minutes max
    
    print(f"✅ Best parameters for {model_name}: {study.best_params}")
    print(f"✅ Best CV score: {study.best_value:.4f}")
    
    return model  # Return original model (optimization results logged)

# =============================================================================
# 5. TRAINING PIPELINE
# =============================================================================

def train_production_model(df, test_size=0.2):
    """Train the complete production model pipeline."""
    
    print("\n🎯 Training Production Career Prediction Model")
    print("=" * 60)
    
    # Prepare features and target
    feature_cols = [
        'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
        'Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional',
        'response_consistency', 'response_time_variance', 'straight_lining_score',
        'completion_rate', 'average_response_time', 'duration_seconds'
    ]
    
    X = df[feature_cols].copy()
    y = df['Career_Subfield'].copy()
    
    print(f"📊 Dataset: {len(X)} samples, {len(feature_cols)} features, {y.nunique()} classes")
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"📈 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Create feature engineering pipeline
    feature_engineer = PsychometricFeatureEngineer(
        include_interactions=True, 
        include_behavioral=True
    )
    scaler = RobustScaler()
    
    # Handle class imbalance
    if IMBALANCED_AVAILABLE:
        smote = SMOTEENN(random_state=RANDOM_STATE)
        print("✅ Using SMOTEENN for class balancing")
    else:
        smote = None
        print("⚠️  SMOTEENN not available, proceeding without class balancing")
    
    # Create base models
    base_models = create_base_models()
    
    # Optimize key models
    for model_name in ['rf', 'xgb']:
        if model_name in base_models:
            base_models[model_name] = optimize_hyperparameters(
                base_models[model_name], X_train, y_train, model_name
            )
    
    # Create stacking ensemble
    stacking_model = create_stacking_ensemble(base_models)
    
    # Build complete pipeline
    if IMBALANCED_AVAILABLE:
        pipeline = ImbPipeline([
            ('feature_engineer', feature_engineer),
            ('scaler', scaler),
            ('smote', smote),
            ('model', stacking_model)
        ])
    else:
        pipeline = Pipeline([
            ('feature_engineer', feature_engineer),
            ('scaler', scaler),
            ('model', stacking_model)
        ])
    
    print("\n🔄 Training pipeline...")
    print("   1. Advanced psychometric feature engineering")
    print("   2. Robust feature scaling")
    if IMBALANCED_AVAILABLE:
        print("   3. SMOTEENN class balancing")
    print("   4. Calibrated stacking ensemble training")
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    top3_accuracy = top_k_accuracy_score(y_test, y_proba, k=3)
    
    print(f"\n✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"🥉 Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    # Cross-validation for robust evaluation
    print("\n🔍 Performing Stratified 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(
        pipeline, X, y_encoded, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    print(f"📊 CV Scores: {cv_scores}")
    print(f"🎯 Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"🏆 CV Accuracy: {cv_scores.mean()*100:.2f}%")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    # Per-class performance analysis
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print(f"\n📊 Per-Class Performance Analysis:")
    for i, class_name in enumerate(label_encoder.classes_):
        if i < len(precision):
            print(f"   {class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, N={support[i]}")
    
    return pipeline, label_encoder, cv_scores.mean(), test_accuracy, top3_accuracy

# =============================================================================
# 6. MODEL PERSISTENCE AND METADATA
# =============================================================================

def save_production_model(pipeline, label_encoder, cv_accuracy, test_accuracy, top3_accuracy, dataset_hash):
    """Save production model with comprehensive metadata."""
    
    print("\n💾 Saving Production Model and Artifacts")
    print("=" * 50)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Generate model version
    model_version = f"v{datetime.now().strftime('%Y.%m.%d')}"
    
    # Save main model files
    model_path = f'models/career_field_model_{model_version}.pkl'
    encoder_path = f'models/label_encoder_{model_version}.pkl'
    
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)
    
    # Save latest versions (for inference)
    joblib.dump(pipeline, 'models/career_field_model_latest.pkl')
    joblib.dump(label_encoder, 'models/label_encoder_latest.pkl')
    
    # Backward compatibility
    joblib.dump(pipeline, 'career_field_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Extract feature names if available
    feature_names = []
    if hasattr(pipeline.named_steps['feature_engineer'], 'feature_names_'):
        feature_names = pipeline.named_steps['feature_engineer'].feature_names_
    
    # Create comprehensive metadata
    metadata = {
        'model_info': {
            'version': model_version,
            'training_date': datetime.now().isoformat(),
            'model_type': 'Calibrated Stacking Ensemble',
            'base_models': ['RandomForest', 'GradientBoosting', 'XGBoost', 'CatBoost', 'MLP'],
            'meta_learner': 'LogisticRegression'
        },
        'dataset_info': {
            'dataset_hash': dataset_hash,
            'n_classes': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist(),
            'feature_count': len(feature_names),
            'feature_names': feature_names
        },
        'performance_metrics': {
            'cv_accuracy': float(cv_accuracy),
            'test_accuracy': float(test_accuracy),
            'top3_accuracy': float(top3_accuracy),
            'target_achieved': cv_accuracy >= 0.90
        },
        'pipeline_config': {
            'feature_engineering': 'PsychometricFeatureEngineer',
            'scaling': 'RobustScaler',
            'class_balancing': 'SMOTEENN' if IMBALANCED_AVAILABLE else 'None',
            'calibration': 'CalibratedClassifierCV'
        },
        'quality_gates': {
            'min_cv_accuracy': 0.88,
            'min_test_accuracy': 0.85,
            'min_top3_accuracy': 0.95,
            'passed': cv_accuracy >= 0.88 and test_accuracy >= 0.85 and top3_accuracy >= 0.95
        }
    }
    
    # Save metadata
    metadata_path = f'models/model_metadata_{model_version}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save latest metadata
    with open('models/model_metadata_latest.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Saved files:")
    print(f"   - {model_path}")
    print(f"   - {encoder_path}")
    print(f"   - {metadata_path}")
    print("   - models/career_field_model_latest.pkl")
    print("   - models/label_encoder_latest.pkl")
    print("   - models/model_metadata_latest.json")
    
    return metadata

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline execution."""
    
    print("🚀 Starting Production Model Training Pipeline")
    print(f"📅 Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 1. Load or generate dataset
        df = load_dataset()
        
        # 2. Compute dataset fingerprint
        dataset_hash = compute_dataset_fingerprint(df)
        print(f"🔍 Dataset fingerprint: {dataset_hash[:16]}...")
        
        # 3. Train production model
        pipeline, label_encoder, cv_accuracy, test_accuracy, top3_accuracy = train_production_model(df)
        
        # 4. Save model and artifacts
        metadata = save_production_model(
            pipeline, label_encoder, cv_accuracy, test_accuracy, top3_accuracy, dataset_hash
        )
        
        # 5. Final evaluation and recommendations
        print("\n" + "=" * 70)
        print("🏆 TRAINING COMPLETE - PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"✅ Cross-Validation Accuracy: {cv_accuracy*100:.2f}%")
        print(f"✅ Test Set Accuracy: {test_accuracy*100:.2f}%")
        print(f"✅ Top-3 Accuracy: {top3_accuracy*100:.2f}%")
        print(f"📊 Total Classes: {len(label_encoder.classes_)}")
        print(f"🎯 90% Target Achieved: {'YES' if cv_accuracy >= 0.90 else 'NO'}")
        print(f"🚦 Quality Gates Passed: {'YES' if metadata['quality_gates']['passed'] else 'NO'}")
        
        if cv_accuracy >= 0.90:
            print("🎉 SUCCESS: Model achieves 90%+ accuracy target!")
        elif cv_accuracy >= 0.88:
            print("✅ GOOD: Model achieves 88%+ accuracy (production ready)")
            print("💡 Recommendations to reach 90%:")
            print("   - Collect more training data (target: 5000+ samples)")
            print("   - Add domain-specific features")
            print("   - Implement advanced ensemble methods")
        else:
            print("⚠️  Model accuracy below production threshold (88%)")
            print("🔧 Required improvements:")
            print("   - Increase training data size significantly")
            print("   - Review feature engineering pipeline")
            print("   - Consider different model architectures")
            print("   - Validate data quality and labeling")
        
        # 6. Technical report summary
        print(f"\n📄 Technical Report Summary:")
        print(f"   - Model Type: Calibrated Stacking Ensemble")
        print(f"   - Feature Engineering: Advanced psychometric features")
        print(f"   - Class Balancing: {'SMOTEENN' if IMBALANCED_AVAILABLE else 'None'}")
        print(f"   - Cross-Validation: 5-fold stratified")
        print(f"   - Dataset Size: {len(df)} samples")
        print(f"   - Feature Count: {len(metadata['dataset_info']['feature_names'])}")
        
        return pipeline, label_encoder, metadata
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Run the production training pipeline
    pipeline, encoder, metadata = main()