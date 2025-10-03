const mongoose = require('mongoose');

const reportSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  answers: {
    Math_Score: { type: Number, required: true },
    Science_Score: { type: Number, required: true },
    English_Score: { type: Number, required: true },
    Extracurricular_Score: { type: Number, required: true },
    Openness: { type: Number, required: true },
    Conscientiousness: { type: Number, required: true },
    Extraversion: { type: Number, required: true },
    Agreeableness: { type: Number, required: true },
    Neuroticism: { type: Number, required: true },
    Percentage: { type: Number, required: true }
  },
  report: {
    top3_careers: {
      type: Map,
      of: Number,
      required: true
    },
    accuracy: { type: Number, default: 95 },
    recommendations: { type: String }
  },
  blockchainHash: {
    type: String,
    default: null
  },
  ipfsCid: {
    type: String,
    default: null
  },
  reportHash: {
    type: String,
    default: null
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'error'],
    default: 'completed'
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('Report', reportSchema);