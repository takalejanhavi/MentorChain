import React from 'react';
import { motion } from 'framer-motion';

interface QuestionCardProps {
  question: string;
  questionNumber: number;
  totalQuestions: number;
  onResponse: (value: number) => void;
  currentResponse?: number;
}

const LIKERT_OPTIONS = [
  { value: 1, label: "Strongly Disagree", color: "from-red-500 to-red-600" },
  { value: 2, label: "Disagree", color: "from-orange-500 to-orange-600" },
  { value: 3, label: "Neutral", color: "from-gray-500 to-gray-600" },
  { value: 4, label: "Agree", color: "from-blue-500 to-blue-600" },
  { value: 5, label: "Strongly Agree", color: "from-green-500 to-green-600" }
];

const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  questionNumber,
  totalQuestions,
  onResponse,
  currentResponse
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10"
    >
      {/* Question Header */}
      <div className="text-center mb-8">
        <div className="text-sm text-gray-400 mb-2">
          Question {questionNumber} of {totalQuestions}
        </div>
        <h2 className="text-2xl font-bold text-white leading-relaxed">
          {question}
        </h2>
      </div>

      {/* Response Options */}
      <div className="space-y-3">
        {LIKERT_OPTIONS.map((option) => (
          <motion.button
            key={option.value}
            onClick={() => onResponse(option.value)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`w-full p-4 rounded-lg text-left transition-all duration-200 flex items-center justify-between ${
              currentResponse === option.value
                ? `bg-gradient-to-r ${option.color} text-white shadow-lg`
                : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
            }`}
          >
            <div className="flex items-center space-x-4">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                currentResponse === option.value ? 'bg-white/20' : 'bg-white/10'
              }`}>
                {option.value}
              </div>
              <span className="font-medium">{option.label}</span>
            </div>
            
            {currentResponse === option.value && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-6 h-6 bg-white rounded-full flex items-center justify-center"
              >
                <div className="w-3 h-3 bg-current rounded-full" />
              </motion.div>
            )}
          </motion.button>
        ))}
      </div>

      {/* Helper Text */}
      <div className="mt-6 text-center text-sm text-gray-400">
        Choose the response that best describes you
      </div>
    </motion.div>
  );
};

export default QuestionCard;