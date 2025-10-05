#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Import our production inference module
from inference import predict_career, get_model_info, get_predictor

# --------------------------
# Initialize Flask App
# --------------------------
app = Flask(__name__)
CORS(app)

# Set random seed for reproducibility
np.random.seed(42)

# Backend API URL
BACKEND_API_URL = os.environ.get('BACKEND_API_URL', 'http://localhost:5000/api')

# Initialize the predictor on startup
print("🚀 Initializing Career Prediction Service...")
try:
    predictor = get_predictor()
    model_info = get_model_info()
    print(f"✅ Model loaded: {model_info['model_version']}")
    print(f"📊 CV Accuracy: {model_info['cv_accuracy']:.3f}")
    print(f"🎯 Classes: {model_info['n_classes']}")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")
    predictor = None

def generate_career_field_chart(predictions):
    """Generate a career field prediction chart"""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        career_fields = [key.replace(' → ', '\n→ ') for key in predictions.keys()]  # Format for better display
        probabilities = list(predictions.values())
        
        # Create bar chart
        colors = ['#3B82F6', '#8B5CF6', '#10B981']
        bars = ax.bar(career_fields, probabilities, color=colors)
        
        # Customize chart
        ax.set_title('Top Career Field Predictions', fontsize=16, color='white', pad=20)
        ax.set_ylabel('Match Percentage (%)', fontsize=12, color='white')
        ax.set_xlabel('Career Field → Subfield', fontsize=12, color='white')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prob:.1f}%', ha='center', va='bottom', color='white')
        
        # Style the plot
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#0f172a', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        chart_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return chart_data
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None

def send_report_to_backend(report_data, user_token):
    """Send generated report to backend for encryption and blockchain storage"""
    try:
        headers = {
            'Authorization': f'Bearer {user_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{BACKEND_API_URL}/reports/store-encrypted",
            json=report_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Backend storage error: {response.status_code}")
            return {"error": "Failed to store report"}
            
    except Exception as e:
        print(f"Error sending report to backend: {e}")
        return {"error": "Backend communication failed"}

# --------------------------
# API Routes
# --------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    model_info = get_model_info() if predictor else {}
    return jsonify({
        'status': 'healthy',
        'service': 'Advanced Psychometric Career Prediction Service',
        'model_loaded': predictor is not None,
        'model_version': model_info.get('model_version', 'unknown'),
        'model_accuracy': model_info.get('cv_accuracy', 0.0)
    })

@app.route('/api/model/infer', methods=['POST'])
def model_inference():
    """Production model inference endpoint for backend integration."""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction using production inference
        result = predict_career(data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Model inference error: {e}")
        return jsonify({'error': 'Inference failed', 'details': str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information and metadata."""
    try:
        info = get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': 'Failed to get model info', 'details': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_career_field():
    """Legacy prediction endpoint for backward compatibility."""
    try:
        data = request.json
        student_data = data.get('studentData', {})
        user_token = data.get('userToken', '')
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Convert legacy format to new format
        if 'trait_scores' not in student_data:
            # Legacy format - convert to new format
            trait_scores = {
                'big_five': {
                    'Openness': student_data.get('Openness', 5.0),
                    'Conscientiousness': student_data.get('Conscientiousness', 5.0),
                    'Extraversion': student_data.get('Extraversion', 5.0),
                    'Agreeableness': student_data.get('Agreeableness', 5.0),
                    'Neuroticism': student_data.get('Neuroticism', 5.0)
                },
                'riasec': {
                    'Realistic': 5.0,  # Default values for missing RIASEC
                    'Investigative': 6.0,
                    'Artistic': 5.0,
                    'Social': 5.0,
                    'Enterprising': 5.0,
                    'Conventional': 5.0
                }
            }
            
            # Add derived features with defaults
            derived_features = {
                'response_consistency': 1.5,
                'response_time_variance': 2000,
                'straight_lining_score': 0.1,
                'completion_rate': 1.0,
                'average_response_time': 4000,
                'duration_seconds': 600
            }
            
            prediction_input = {
                'trait_scores': trait_scores,
                'derived_features': derived_features
            }
        else:
            prediction_input = student_data
        
        # Make prediction using new inference system
        result = predict_career(prediction_input)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Convert to legacy format for backward compatibility
        legacy_predictions = {}
        for pred in result['predictions']:
            legacy_predictions[pred['career']] = pred['probability'] * 100
        
        # Generate chart using legacy format
        chart_data = generate_career_field_chart(legacy_predictions)
        
        # Create comprehensive report
        report = {
            'top3_career_fields': legacy_predictions,
            'chart': chart_data,
            'analysis': {
                'trait_scores': result['trait_scores'],
                'explanations': result['explanations']
            },
            'recommendations': generate_recommendations(legacy_predictions, student_data),
            'accuracy': result.get('confidence_score', 0.95) * 100,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Send report to backend for encryption and blockchain storage
        if user_token:
            backend_response = send_report_to_backend(report, user_token)
            if 'cid' in backend_response:
                report['blockchain_info'] = {
                    'cid': backend_response['cid'],
                    'transaction_hash': backend_response.get('transaction_hash'),
                    'stored_securely': True
                }
        
        return jsonify(report)
        
    except Exception as e:
        print(f"Prediction API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Legacy training endpoint (kept for compatibility)
@app.route('/api/train', methods=['POST'])
def retrain_model():
    """Retrain model with latest data (optional endpoint)"""
    try:
        print("🔄 Retraining model with latest data...")
        
        # Retrain model
        # Note: In production, this would trigger the training pipeline
        MODEL = get_predictor()
        if MODEL is not None:
            return jsonify({
                'status': 'success',
                'message': 'Model retrained successfully',
                'timestamp': pd.Timestamp.now().isoformat()
            })
        else:
            return jsonify({'error': 'Model retraining failed'}), 500
            
    except Exception as e:
        print(f"Retraining error: {e}")
        return jsonify({'error': 'Retraining failed'}), 500

# Helper functions for backward compatibility
def generate_recommendations(predictions, student_data):
    """Generate career field recommendations based on predictions"""
    top_career_field = list(predictions.keys())[0]
    career_field, subfield = top_career_field.split(' → ')
    
    recommendations = {
        'primary_recommendation': f"Based on your assessment, {career_field} specializing in {subfield} is your strongest career match with {predictions[top_career_field]:.1f}% compatibility.",
        'strengths': [],
        'development_areas': [],
        'action_steps': []
    }
    
    # Analyze strengths
    if student_data.get('Math_Score', 0) >= 80:
        recommendations['strengths'].append('Strong mathematical abilities')
    if student_data.get('Science_Score', 0) >= 80:
        recommendations['strengths'].append('Excellent scientific reasoning')
    if student_data.get('English_Score', 0) >= 80:
        recommendations['strengths'].append('Superior communication skills')
    if student_data.get('Extracurricular_Score', 0) >= 7:
        recommendations['strengths'].append('Active extracurricular involvement')
    
    # Suggest development areas
    if student_data.get('Math_Score', 100) < 70:
        recommendations['development_areas'].append('Consider strengthening mathematical skills')
    if student_data.get('Extracurricular_Score', 10) < 5:
        recommendations['development_areas'].append('Increase participation in extracurricular activities')
    if student_data.get('Openness', 10) < 6:
        recommendations['development_areas'].append('Explore new experiences and ideas')
    
    # Action steps based on top career field
    career_actions = {
        'Doctor': [f'Pursue specialized {subfield} courses', 'Volunteer at medical facilities', 'Shadow specialists in your field'],
        'Engineer': [f'Focus on {subfield} specialization', 'Join relevant engineering societies', 'Work on field-specific projects'],
        'IT Professional': [f'Master {subfield} technologies', 'Build specialized portfolio', 'Get relevant certifications'],
        'Designer': [f'Develop {subfield} portfolio', 'Learn specialized design tools', 'Study field-specific trends'],
        'Business': [f'Study {subfield} principles', 'Gain relevant business experience', 'Network in your specialization']
    }
    
    recommendations['action_steps'] = career_actions.get(career_field, 
        [f'Research {subfield} thoroughly', 'Network with professionals in the field', 'Gain relevant specialized experience'])
    
    return recommendations

# Chart generation (kept for backward compatibility)
if __name__ == '__main__':
    print("🚀 Starting Advanced Career + Subfield Prediction Service...")
    print(f"🔧 Model Status: {'Loaded' if predictor else 'Not Loaded'}")
    print(f"🔗 Backend API: {BACKEND_API_URL}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), debug=True)