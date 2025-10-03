import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Send, Brain } from 'lucide-react';
import { api, modelApi } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import ReportGenerator from '../components/ReportGenerator';

interface Question {
  id: string;
  type: 'range' | 'select' | 'number';
  question: string;
  field: string;
  options?: string[];
  min?: number;
  max?: number;
  description?: string;
}
interface Report {
  top3_career_fields?: Record<string, number>;
  chart?: string;
  timestamp?: string;
  blockchain_info?: any;
}


const TakeTest = () => {
  const { user } = useAuth();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [report, setReport] = useState<Report>({});
  const [showStorage, setShowStorage] = useState(false);

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      // Define the questions based on the Python model requirements
      const testQuestions: Question[] = [
        {
          id: '1',
          type: 'number',
          field: 'Math_Score',
          question: 'What is your Math score?',
          description: 'Enter your math score out of 100',
          min: 0,
          max: 100
        },
        {
          id: '2',
          type: 'number',
          field: 'Science_Score',
          question: 'What is your Science score?',
          description: 'Enter your science score out of 100',
          min: 0,
          max: 100
        },
        {
          id: '3',
          type: 'number',
          field: 'English_Score',
          question: 'What is your English score?',
          description: 'Enter your English score out of 100',
          min: 0,
          max: 100
        },
        {
          id: '4',
          type: 'range',
          field: 'Extracurricular_Score',
          question: 'Rate your extracurricular involvement',
          description: 'How actively are you involved in extracurricular activities?',
          min: 1,
          max: 10
        },
        {
          id: '5',
          type: 'range',
          field: 'Openness',
          question: 'Rate your openness to new experiences',
          description: 'How open are you to trying new things and exploring new ideas?',
          min: 1,
          max: 10
        },
        {
          id: '6',
          type: 'range',
          field: 'Conscientiousness',
          question: 'Rate your conscientiousness',
          description: 'How organized, disciplined, and goal-oriented are you?',
          min: 1,
          max: 10
        },
        {
          id: '7',
          type: 'range',
          field: 'Extraversion',
          question: 'Rate your extraversion',
          description: 'How outgoing and social are you?',
          min: 1,
          max: 10
        },
        {
          id: '8',
          type: 'range',
          field: 'Agreeableness',
          question: 'Rate your agreeableness',
          description: 'How cooperative and trusting are you with others?',
          min: 1,
          max: 10
        },
        {
          id: '9',
          type: 'range',
          field: 'Neuroticism',
          question: 'Rate your emotional stability',
          description: 'How well do you handle stress and emotional challenges? (1 = Very stable, 10 = Very anxious)',
          min: 1,
          max: 10
        }
      ];

      setQuestions(testQuestions);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching questions:', error);
      setLoading(false);
    }
  };

  const handleAnswerChange = (value: any) => {
    setAnswers(prev => ({
      ...prev,
      [questions[currentQuestion].field]: value
    }));
  };

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
    }
  };

  const prevQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
    }
  };

  const submitTest = async () => {
    setSubmitting(true);
    
    try {
      // Calculate percentage as weighted average
      const mathScore = parseInt(answers.Math_Score) || 0;
      const scienceScore = parseInt(answers.Science_Score) || 0;
      const englishScore = parseInt(answers.English_Score) || 0;
      const extraScore = parseInt(answers.Extracurricular_Score) || 0;
      
      const percentage = Math.round(
        (mathScore * 0.3 + scienceScore * 0.3 + englishScore * 0.2 + extraScore * 2 * 0.2)
      );
      
      const studentData = {
        ...answers,
        Percentage: percentage
      };

      // Get user token for backend communication
      const token = localStorage.getItem('token');
      // Send to Python model service
      const modelResponse = await modelApi.post('/predict', { 
        studentData,
        userToken: token 
      });
      
      setReport(modelResponse.data);
      setCompleted(true);
      setShowStorage(true);
    } catch (error) {
      console.error('Error submitting test:', error);
    }
    
    setSubmitting(false);
  };

const handleReportStored = (result: any) => {
  setShowStorage(false);

  setReport(prev => ({
    ...prev,           // prev is always a Report object
    blockchain_info: result
  }));
};

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">Loading assessment...</div>
      </div>
    );
  }

  if (completed) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-4xl mx-auto"
      >
        <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-white mb-2">Assessment Complete!</h2>
            <p className="text-gray-300">Your advanced career + subfield report has been generated</p>
          </div>

          {report && (
            <div className="space-y-6">
              <h3 className="text-xl font-semibold text-white">Top Career Field Recommendations</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(report.top3_career_fields || {}).map(([careerField, probability]: [string, any], index) => (
                  <div key={index} className="bg-white/10 rounded-lg p-4 border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-white text-sm">{careerField.replace(' → ', ' → ')}</h4>
                      <span className="text-sm text-gray-300">#{index + 1}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${probability}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-300">{probability.toFixed(1)}% match</p>
                  </div>
                ))}
              </div>
              
              {report.blockchain_info && (
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                  <h4 className="text-green-400 font-medium mb-2">🔒 Securely Stored</h4>
                  <p className="text-sm text-gray-300">
                    Your report has been encrypted and stored on IPFS with blockchain verification.
                  </p>
                </div>
              )}
            </div>
          )}

          <div className="mt-8 text-center">
            <button
              onClick={() => window.location.href = '/reports'}
              className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:shadow-lg transition-all duration-200"
            >
              View Detailed Report
            </button>
          </div>
        </div>
        
        {showStorage && report && (
          <ReportGenerator 
            reportData={report} 
            onReportStored={handleReportStored}
          />
        )}
      </motion.div>
    );
  }

  const currentQ = questions[currentQuestion];
  const currentAnswer = answers[currentQ?.field];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="max-w-3xl mx-auto"
    >
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-300">Question {currentQuestion + 1} of {questions.length}</span>
          <span className="text-sm text-gray-300">{Math.round(progress)}% Complete</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Question Card */}
      <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
        <h2 className="text-2xl font-bold text-white mb-4">{currentQ?.question}</h2>
        {currentQ?.description && (
          <p className="text-gray-300 mb-6">{currentQ.description}</p>
        )}

        {/* Answer Input */}
        <div className="mb-8">
          {currentQ?.type === 'number' && (
            <div>
              <input
                type="number"
                min={currentQ.min}
                max={currentQ.max}
                value={currentAnswer || ''}
                onChange={(e) => handleAnswerChange(parseInt(e.target.value))}
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder={`Enter value (${currentQ.min}-${currentQ.max})`}
              />
            </div>
          )}

          {currentQ?.type === 'range' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-gray-300">{currentQ.min}</span>
                <span className="text-white font-semibold text-lg">{currentAnswer || currentQ.min}</span>
                <span className="text-gray-300">{currentQ.max}</span>
              </div>
              <input
                type="range"
                min={currentQ.min}
                max={currentQ.max}
                value={currentAnswer || currentQ.min}
                onChange={(e) => handleAnswerChange(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
          )}

          {currentQ?.type === 'select' && currentQ.options && (
            <div className="grid grid-cols-1 gap-3">
              {currentQ.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleAnswerChange(option)}
                  className={`p-4 rounded-lg text-left transition-all duration-200 ${
                    currentAnswer === option
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={prevQuestion}
            disabled={currentQuestion === 0}
            className="flex items-center space-x-2 px-6 py-3 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-5 h-5" />
            <span>Previous</span>
          </button>

          {currentQuestion === questions.length - 1 ? (
            <button
              onClick={submitTest}
              disabled={submitting || !currentAnswer}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <div className="w-5 h-5 border-2 border-white/20 border-top-white rounded-full animate-spin"></div>
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>Submit Assessment</span>
            </button>
          ) : (
            <button
              onClick={nextQuestion}
              disabled={!currentAnswer}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>Next</span>
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default TakeTest;