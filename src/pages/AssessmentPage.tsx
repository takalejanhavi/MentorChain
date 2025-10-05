import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Save, Brain, Clock, CheckCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../services/api';

interface Question {
  id: number;
  text: string;
  trait: string;
  reverse_scored: boolean;
}

interface AssessmentResponse {
  question_id: number;
  response: number;
  response_time_ms: number;
}

const QUESTIONS: Question[] = [
  { id: 1, text: "I enjoy exploring new ideas and concepts", trait: "Openness", reverse_scored: false },
  { id: 2, text: "I am curious about how things work", trait: "Investigative", reverse_scored: false },
  { id: 3, text: "I like to think about abstract theories", trait: "Conventional", reverse_scored: false },
  { id: 4, text: "I enjoy creative activities like art or writing", trait: "Artistic", reverse_scored: false },
  { id: 5, text: "I prefer routine tasks over novel challenges", trait: "Openness", reverse_scored: true },
  { id: 6, text: "I am interested in learning about different cultures", trait: "Social", reverse_scored: false },
  { id: 7, text: "I enjoy philosophical discussions", trait: "Openness", reverse_scored: false },
  { id: 8, text: "I am always prepared for meetings and appointments", trait: "Conscientiousness", reverse_scored: false },
  { id: 9, text: "I pay attention to details in my work", trait: "Investigative", reverse_scored: false },
  { id: 10, text: "I follow through on my commitments", trait: "Conventional", reverse_scored: false },
  { id: 11, text: "I prefer to plan ahead rather than be spontaneous", trait: "Conscientiousness", reverse_scored: false },
  { id: 12, text: "I often leave tasks unfinished", trait: "Conscientiousness", reverse_scored: true },
  { id: 13, text: "I am organized in my approach to work", trait: "Social", reverse_scored: false },
  { id: 14, text: "I set high standards for myself", trait: "Conscientiousness", reverse_scored: false },
  { id: 15, text: "I enjoy being the center of attention", trait: "Extraversion", reverse_scored: false },
  { id: 16, text: "I feel comfortable speaking in front of groups", trait: "Investigative", reverse_scored: false },
  { id: 17, text: "I prefer working in teams rather than alone", trait: "Conventional", reverse_scored: false },
  { id: 18, text: "I am energized by social interactions", trait: "Artistic", reverse_scored: false },
  { id: 19, text: "I tend to be quiet in social situations", trait: "Extraversion", reverse_scored: true },
  { id: 20, text: "I enjoy meeting new people", trait: "Social", reverse_scored: false },
  { id: 21, text: "I am comfortable taking leadership roles", trait: "Extraversion", reverse_scored: false },
  { id: 22, text: "I try to be helpful to others", trait: "Enterprising", reverse_scored: false },
  { id: 23, text: "I trust people's intentions are generally good", trait: "Investigative", reverse_scored: false },
  { id: 24, text: "I avoid conflicts when possible", trait: "Conventional", reverse_scored: false },
  { id: 25, text: "I am sympathetic to others' problems", trait: "Artistic", reverse_scored: false },
  { id: 26, text: "I tend to be critical of others", trait: "Agreeableness", reverse_scored: true },
  { id: 27, text: "I enjoy cooperating with others", trait: "Social", reverse_scored: false },
  { id: 28, text: "I am forgiving when others make mistakes", trait: "Agreeableness", reverse_scored: false },
  { id: 29, text: "I often worry about things that might go wrong", trait: "Enterprising", reverse_scored: false },
  { id: 30, text: "I get stressed easily under pressure", trait: "Investigative", reverse_scored: false },
  { id: 31, text: "I am generally calm and relaxed", trait: "Neuroticism", reverse_scored: true },
  { id: 32, text: "I experience mood swings frequently", trait: "Artistic", reverse_scored: false },
  { id: 33, text: "I feel anxious in uncertain situations", trait: "Realistic", reverse_scored: false },
  { id: 34, text: "I recover quickly from setbacks", trait: "Social", reverse_scored: false },
  { id: 35, text: "I tend to feel overwhelmed by responsibilities", trait: "Neuroticism", reverse_scored: false }
];

const LIKERT_SCALE = [
  { value: 1, label: "Strongly Disagree" },
  { value: 2, label: "Disagree" },
  { value: 3, label: "Neutral" },
  { value: 4, label: "Agree" },
  { value: 5, label: "Strongly Agree" }
];

const AssessmentPage: React.FC = () => {
  const { user } = useAuth();
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [responses, setResponses] = useState<Record<number, AssessmentResponse>>({});
  const [startTime, setStartTime] = useState<number>(Date.now());
  const [questionStartTime, setQuestionStartTime] = useState<number>(Date.now());
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [savedProgress, setSavedProgress] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    // Load saved progress if exists
    const savedData = localStorage.getItem(`assessment_progress_${user?.id}`);
    if (savedData) {
      const { responses: savedResponses, currentQuestion: savedQuestion } = JSON.parse(savedData);
      setResponses(savedResponses);
      setCurrentQuestion(savedQuestion);
    }
    setQuestionStartTime(Date.now());
  }, [user?.id]);

  const handleResponse = (value: number) => {
    const responseTime = Date.now() - questionStartTime;
    const newResponse: AssessmentResponse = {
      question_id: QUESTIONS[currentQuestion].id,
      response: value,
      response_time_ms: responseTime
    };

    setResponses(prev => ({
      ...prev,
      [QUESTIONS[currentQuestion].id]: newResponse
    }));

    // Auto-advance to next question after short delay
    setTimeout(() => {
      if (currentQuestion < QUESTIONS.length - 1) {
        setCurrentQuestion(prev => prev + 1);
        setQuestionStartTime(Date.now());
      }
    }, 300);
  };

  const saveProgress = async () => {
    const progressData = {
      responses,
      currentQuestion,
      sessionId,
      timestamp: Date.now()
    };
    
    localStorage.setItem(`assessment_progress_${user?.id}`, JSON.stringify(progressData));
    setSavedProgress(true);
    setTimeout(() => setSavedProgress(false), 2000);
  };

  const submitAssessment = async () => {
    setIsSubmitting(true);
    
    try {
      const assessmentData = {
        session_id: sessionId,
        responses: Object.values(responses),
        duration_seconds: Math.floor((Date.now() - startTime) / 1000),
        device_type: /Mobile|Android|iPhone|iPad/.test(navigator.userAgent) ? 'mobile' : 'desktop',
        timestamp: new Date().toISOString()
      };

      const response = await api.post('/assessment/submit', assessmentData);
      
      // Clear saved progress
      localStorage.removeItem(`assessment_progress_${user?.id}`);
      
      setIsCompleted(true);
      
      // Redirect to results after short delay
      setTimeout(() => {
        window.location.href = `/report/${response.data.assessment_id}`;
      }, 2000);
      
    } catch (error) {
      console.error('Error submitting assessment:', error);
      alert('Error submitting assessment. Please try again.');
    }
    
    setIsSubmitting(false);
  };

  const progress = ((currentQuestion + 1) / QUESTIONS.length) * 100;
  const currentQ = QUESTIONS[currentQuestion];
  const currentResponse = responses[currentQ?.id];

  if (isCompleted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">Assessment Complete!</h2>
          <p className="text-gray-300">Generating your personalized career report...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Brain className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold text-white">Career Assessment</h1>
          </div>
          <p className="text-gray-300">
            Answer honestly to get the most accurate career recommendations
          </p>
        </motion.div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">
              Question {currentQuestion + 1} of {QUESTIONS.length}
            </span>
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">
                ~{Math.ceil((QUESTIONS.length - currentQuestion - 1) * 0.3)} min remaining
              </span>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Question Card */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentQuestion}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 mb-8"
          >
            <h2 className="text-2xl font-bold text-white mb-8 text-center">
              {currentQ?.text}
            </h2>

            {/* Likert Scale */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {LIKERT_SCALE.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handleResponse(option.value)}
                  className={`p-4 rounded-lg text-center transition-all duration-200 ${
                    currentResponse?.response === option.value
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
                  }`}
                >
                  <div className="font-semibold mb-1">{option.value}</div>
                  <div className="text-sm">{option.label}</div>
                </button>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => {
              if (currentQuestion > 0) {
                setCurrentQuestion(prev => prev - 1);
                setQuestionStartTime(Date.now());
              }
            }}
            disabled={currentQuestion === 0}
            className="flex items-center space-x-2 px-6 py-3 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-5 h-5" />
            <span>Previous</span>
          </button>

          <div className="flex items-center space-x-4">
            <button
              onClick={saveProgress}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                savedProgress 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Save className="w-4 h-4" />
              <span>{savedProgress ? 'Saved!' : 'Save Progress'}</span>
            </button>

            {currentQuestion === QUESTIONS.length - 1 && Object.keys(responses).length === QUESTIONS.length ? (
              <button
                onClick={submitAssessment}
                disabled={isSubmitting}
                className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:shadow-lg transition-all duration-200 disabled:opacity-50"
              >
                {isSubmitting ? (
                  <div className="w-5 h-5 border-2 border-white/20 border-top-white rounded-full animate-spin" />
                ) : (
                  <CheckCircle className="w-5 h-5" />
                )}
                <span>Complete Assessment</span>
              </button>
            ) : (
              <button
                onClick={() => {
                  if (currentQuestion < QUESTIONS.length - 1) {
                    setCurrentQuestion(prev => prev + 1);
                    setQuestionStartTime(Date.now());
                  }
                }}
                disabled={currentQuestion === QUESTIONS.length - 1}
                className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200 disabled:opacity-50"
              >
                <span>Next</span>
                <ChevronRight className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AssessmentPage;