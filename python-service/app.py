#!/usr/bin/env python3
"""
Complete Career Prediction System - Training, Evaluation & Prediction
Uses existing student_training_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CAREER PREDICTION SYSTEM - COMPLETE PIPELINE")
print("="*70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/6] Loading training data...")
try:
    df = pd.read_csv('C:/Users/DELL/Downloads/Mentor-Chain/python-service/student_training_data.csv')
    print(f"✅ Loaded {len(df):,} samples")
    print(f"   - Features: {df.shape[1] - 2} (excluding Career_Field & Subfield)")
    print(f"   - Career Fields: {df['Career_Field'].nunique()}")
    print(f"   - Subfields: {df['Subfield'].nunique()}")
except FileNotFoundError:
    print("❌ Error: student_training_data.csv not found!")
    print("   Please run the data generation script first.")
    exit(1)

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
print("\n[2/6] Preparing data...")

# Define feature columns
feature_cols = ['Math_Score', 'Science_Score', 'English_Score', 
                'Extracurricular_Score', 'Openness', 'Conscientiousness',
                'Extraversion', 'Agreeableness', 'Neuroticism', 'Percentage']

X = df[feature_cols].values
y_career = df['Career_Field'].values
y_subfield = df['Subfield'].values

# Encode labels
le_career = LabelEncoder()
le_subfield = LabelEncoder()

y_career_encoded = le_career.fit_transform(y_career)
y_subfield_encoded = le_subfield.fit_transform(y_subfield)

print(f"✅ Features prepared: {X.shape}")
print(f"   - Career classes: {len(le_career.classes_)}")
print(f"   - Subfield classes: {len(le_subfield.classes_)}")

# Split data
X_train, X_test, y_career_train, y_career_test, y_subfield_train, y_subfield_test = train_test_split(
    X, y_career_encoded, y_subfield_encoded, test_size=0.2, random_state=42, stratify=y_career_encoded
)

print(f"   - Training samples: {len(X_train):,}")
print(f"   - Testing samples: {len(X_test):,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 3: TRAIN CAREER FIELD MODEL
# ============================================================
print("\n[3/6] Training Career Field model...")
print("   Algorithm: Random Forest (optimized for high accuracy)")

career_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

career_model.fit(X_train_scaled, y_career_train)
print("✅ Career Field model trained")

# Evaluate
career_pred = career_model.predict(X_test_scaled)
career_accuracy = accuracy_score(y_career_test, career_pred)
print(f"   📊 Accuracy: {career_accuracy*100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(career_model, X_train_scaled, y_career_train, cv=5)
print(f"   📊 Cross-Val: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

# ============================================================
# STEP 4: TRAIN SUBFIELD MODEL
# ============================================================
print("\n[4/6] Training Subfield model...")
print("   Algorithm: Gradient Boosting (handles complex patterns)")

subfield_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=15,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)

subfield_model.fit(X_train_scaled, y_subfield_train)
print("✅ Subfield model trained")

# Evaluate
subfield_pred = subfield_model.predict(X_test_scaled)
subfield_accuracy = accuracy_score(y_subfield_test, subfield_pred)
print(f"   📊 Accuracy: {subfield_accuracy*100:.2f}%")

# ============================================================
# STEP 5: DETAILED EVALUATION
# ============================================================
print("\n[5/6] Detailed Evaluation...")

print("\n📈 CAREER FIELD PERFORMANCE:")
print("-" * 70)
career_report = classification_report(
    y_career_test, career_pred, 
    target_names=le_career.classes_, 
    zero_division=0
)
print(career_report)

print("\n📈 SUBFIELD PERFORMANCE (Top 10 by support):")
print("-" * 70)
subfield_report = classification_report(
    y_subfield_test, subfield_pred,
    target_names=le_subfield.classes_,
    output_dict=True,
    zero_division=0
)

# Sort by support and show top 10
subfield_items = [(name, metrics) for name, metrics in subfield_report.items() 
                  if isinstance(metrics, dict) and 'support' in metrics]
subfield_items.sort(key=lambda x: x[1]['support'], reverse=True)

print(f"{'Subfield':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 70)
for name, metrics in subfield_items[:10]:
    print(f"{name:<30} {metrics['precision']:>10.2f} {metrics['recall']:>10.2f} "
          f"{metrics['f1-score']:>10.2f} {metrics['support']:>10.0f}")

# Feature importance
print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 70)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Career_Importance': career_model.feature_importances_,
    'Subfield_Importance': subfield_model.feature_importances_
}).sort_values('Career_Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['Feature']:<25} Career: {row['Career_Importance']:.4f}  "
          f"Subfield: {row['Subfield_Importance']:.4f}")

# ============================================================
# STEP 6: SAVE MODELS
# ============================================================
print("\n[6/6] Saving models...")

joblib.dump(career_model, 'career_field_model.pkl')
joblib.dump(subfield_model, 'subfield_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_career, 'label_encoder_career.pkl')
joblib.dump(le_subfield, 'label_encoder_subfield.pkl')

print("✅ Models saved:")
print("   - career_field_model.pkl")
print("   - subfield_model.pkl")
print("   - scaler.pkl")
print("   - label_encoder_career.pkl")
print("   - label_encoder_subfield.pkl")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
print("\n" + "="*70)
print("PREDICTION SYSTEM READY")
print("="*70)

def predict_career(student_data):
    """
    Predict career and subfield for a student
    
    Parameters:
    student_data: dict with keys:
        Math_Score, Science_Score, English_Score, Extracurricular_Score,
        Openness, Conscientiousness, Extraversion, Agreeableness, 
        Neuroticism, Percentage
    
    Returns:
    dict with predicted career, subfield, and probabilities
    """
    # Load models
    career_model = joblib.load('career_field_model.pkl')
    subfield_model = joblib.load('subfield_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_career = joblib.load('label_encoder_career.pkl')
    le_subfield = joblib.load('label_encoder_subfield.pkl')
    
    # Prepare input
    feature_order = ['Math_Score', 'Science_Score', 'English_Score',
                     'Extracurricular_Score', 'Openness', 'Conscientiousness',
                     'Extraversion', 'Agreeableness', 'Neuroticism', 'Percentage']
    
    X_input = np.array([[student_data[f] for f in feature_order]])
    X_scaled = scaler.transform(X_input)
    
    # Predict career
    career_pred = career_model.predict(X_scaled)[0]
    career_proba = career_model.predict_proba(X_scaled)[0]
    career_name = le_career.inverse_transform([career_pred])[0]
    
    # Get top 3 careers
    top_career_indices = np.argsort(career_proba)[-3:][::-1]
    top_careers = [(le_career.inverse_transform([idx])[0], career_proba[idx]) 
                   for idx in top_career_indices]
    
    # Predict subfield
    subfield_pred = subfield_model.predict(X_scaled)[0]
    subfield_proba = subfield_model.predict_proba(X_scaled)[0]
    subfield_name = le_subfield.inverse_transform([subfield_pred])[0]
    
    # Get top 3 subfields
    top_subfield_indices = np.argsort(subfield_proba)[-3:][::-1]
    top_subfields = [(le_subfield.inverse_transform([idx])[0], subfield_proba[idx])
                     for idx in top_subfield_indices]
    
    return {
        'career': career_name,
        'career_confidence': career_proba[career_pred],
        'top_careers': top_careers,
        'subfield': subfield_name,
        'subfield_confidence': subfield_proba[subfield_pred],
        'top_subfields': top_subfields
    }

# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================
print("\n📝 EXAMPLE PREDICTIONS:")
print("="*70)

# Example 1: High math/science student
example1 = {
    'Math_Score': 92, 'Science_Score': 88, 'English_Score': 75,
    'Extracurricular_Score': 70, 'Openness': 85, 'Conscientiousness': 80,
    'Extraversion': 60, 'Agreeableness': 65, 'Neuroticism': 35,
    'Percentage': 85
}

print("\n🎓 Student 1: High Math/Science, High Openness")
result1 = predict_career(example1)
print(f"   Career: {result1['career']} ({result1['career_confidence']*100:.1f}%)")
print(f"   Subfield: {result1['subfield']} ({result1['subfield_confidence']*100:.1f}%)")
print(f"   Top careers: {', '.join([f'{c} ({p*100:.1f}%)' for c, p in result1['top_careers']])}")

# Example 2: People-oriented student
example2 = {
    'Math_Score': 70, 'Science_Score': 75, 'English_Score': 88,
    'Extracurricular_Score': 82, 'Openness': 78, 'Conscientiousness': 85,
    'Extraversion': 90, 'Agreeableness': 92, 'Neuroticism': 25,
    'Percentage': 80
}

print("\n🎓 Student 2: High English, Very High People Skills")
result2 = predict_career(example2)
print(f"   Career: {result2['career']} ({result2['career_confidence']*100:.1f}%)")
print(f"   Subfield: {result2['subfield']} ({result2['subfield_confidence']*100:.1f}%)")
print(f"   Top careers: {', '.join([f'{c} ({p*100:.1f}%)' for c, p in result2['top_careers']])}")

# Example 3: Creative student
example3 = {
    'Math_Score': 65, 'Science_Score': 60, 'English_Score': 82,
    'Extracurricular_Score': 90, 'Openness': 95, 'Conscientiousness': 70,
    'Extraversion': 75, 'Agreeableness': 72, 'Neuroticism': 45,
    'Percentage': 75
}

print("\n🎓 Student 3: High Creativity, Strong Extracurricular")
result3 = predict_career(example3)
print(f"   Career: {result3['career']} ({result3['career_confidence']*100:.1f}%)")
print(f"   Subfield: {result3['subfield']} ({result3['subfield_confidence']*100:.1f}%)")
print(f"   Top careers: {', '.join([f'{c} ({p*100:.1f}%)' for c, p in result3['top_careers']])}")

print("\n" + "="*70)
print("✅ SYSTEM COMPLETE - Ready for predictions!")
print("="*70)
print("\nTo use the prediction function:")
print("  result = predict_career(your_student_data)")
print("\nModel files are saved and ready for deployment.")
print("="*70)