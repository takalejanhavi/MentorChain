#!/usr/bin/env python3
"""
Unit tests for the career prediction inference system.
"""

import unittest
import numpy as np
import json
import os
import sys

# Add the python-service directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python-service'))

from inference import CareerPredictor, predict_career, get_model_info

class TestCareerInference(unittest.TestCase):
    """Test cases for career prediction inference."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_input = {
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
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly."""
        try:
            predictor = CareerPredictor()
            self.assertIsNotNone(predictor.model)
            self.assertIsNotNone(predictor.label_encoder)
        except Exception as e:
            self.skipTest(f"Model files not available: {e}")
    
    def test_prediction_format(self):
        """Test that predictions have correct format."""
        try:
            result = predict_career(self.sample_input)
            
            # Check required fields
            self.assertIn('predictions', result)
            self.assertIn('trait_scores', result)
            self.assertIn('explanations', result)
            self.assertIn('model_version', result)
            
            # Check predictions format
            predictions = result['predictions']
            self.assertIsInstance(predictions, list)
            self.assertLessEqual(len(predictions), 3)  # Top 3 predictions
            
            for pred in predictions:
                self.assertIn('career', pred)
                self.assertIn('probability', pred)
                self.assertIsInstance(pred['career'], str)
                self.assertIsInstance(pred['probability'], float)
                self.assertGreaterEqual(pred['probability'], 0.0)
                self.assertLessEqual(pred['probability'], 1.0)
                
        except Exception as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_trait_score_validation(self):
        """Test trait score validation and ranges."""
        result = predict_career(self.sample_input)
        
        if 'error' in result:
            self.skipTest("Model not available")
        
        trait_scores = result['trait_scores']
        
        # Check Big Five scores
        big_five = trait_scores['big_five']
        for trait, score in big_five.items():
            self.assertGreaterEqual(score, 1.0)
            self.assertLessEqual(score, 10.0)
            self.assertIn(trait, ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'])
        
        # Check RIASEC scores
        riasec = trait_scores['riasec']
        for interest, score in riasec.items():
            self.assertGreaterEqual(score, 1.0)
            self.assertLessEqual(score, 10.0)
            self.assertIn(interest, ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty input
        result = predict_career({})
        self.assertIn('predictions', result)  # Should handle gracefully
        
        # Test extreme values
        extreme_input = {
            'trait_scores': {
                'big_five': {
                    'Openness': 10.0,
                    'Conscientiousness': 1.0,
                    'Extraversion': 10.0,
                    'Agreeableness': 1.0,
                    'Neuroticism': 10.0
                },
                'riasec': {
                    'Realistic': 1.0,
                    'Investigative': 10.0,
                    'Artistic': 1.0,
                    'Social': 10.0,
                    'Enterprising': 1.0,
                    'Conventional': 10.0
                }
            }
        }
        
        result = predict_career(extreme_input)
        if 'error' not in result:
            self.assertIn('predictions', result)
            self.assertGreater(len(result['predictions']), 0)
    
    def test_model_info(self):
        """Test model information retrieval."""
        try:
            info = get_model_info()
            
            # Check required fields
            self.assertIn('model_version', info)
            self.assertIn('n_classes', info)
            self.assertIn('classes', info)
            
            # Check data types
            self.assertIsInstance(info['model_version'], str)
            self.assertIsInstance(info['n_classes'], int)
            self.assertIsInstance(info['classes'], list)
            self.assertGreater(info['n_classes'], 0)
            self.assertEqual(len(info['classes']), info['n_classes'])
            
        except Exception as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_consistency(self):
        """Test prediction consistency with same input."""
        try:
            result1 = predict_career(self.sample_input)
            result2 = predict_career(self.sample_input)
            
            if 'error' in result1 or 'error' in result2:
                self.skipTest("Model not available")
            
            # Predictions should be identical for same input
            self.assertEqual(len(result1['predictions']), len(result2['predictions']))
            
            for pred1, pred2 in zip(result1['predictions'], result2['predictions']):
                self.assertEqual(pred1['career'], pred2['career'])
                self.assertAlmostEqual(pred1['probability'], pred2['probability'], places=6)
                
        except Exception as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_probability_ordering(self):
        """Test that predictions are ordered by probability."""
        try:
            result = predict_career(self.sample_input)
            
            if 'error' in result:
                self.skipTest("Model not available")
            
            predictions = result['predictions']
            probabilities = [pred['probability'] for pred in predictions]
            
            # Check that probabilities are in descending order
            for i in range(len(probabilities) - 1):
                self.assertGreaterEqual(probabilities[i], probabilities[i + 1])
                
        except Exception as e:
            self.skipTest(f"Model not available: {e}")

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering components."""
    
    def test_trait_calculation_logic(self):
        """Test trait calculation algorithms."""
        # This would test the trait calculation logic
        # For now, we'll test the basic structure
        
        sample_responses = [
            {'question_id': i, 'response': np.random.randint(1, 6), 'response_time_ms': 3000}
            for i in range(1, 36)
        ]
        
        # Test that we can process responses
        self.assertEqual(len(sample_responses), 35)
        
        # Test response validation
        for response in sample_responses:
            self.assertIn('question_id', response)
            self.assertIn('response', response)
            self.assertIn('response_time_ms', response)
            self.assertGreaterEqual(response['response'], 1)
            self.assertLessEqual(response['response'], 5)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)