const mongoose = require('mongoose');

const assessmentSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  sessionId: {
    type: String,
    required: true,
    unique: true
  },
  responses: [{
    question_id: {
      type: Number,
      required: true
    },
    response: {
      type: Number,
      required: true,
      min: 1,
      max: 5
    },
    response_time_ms: {
      type: Number,
      required: true
    }
  }],
  traitScores: {
    big_five: {
      Openness: { type: Number, required: true },
      Conscientiousness: { type: Number, required: true },
      Extraversion: { type: Number, required: true },
      Agreeableness: { type: Number, required: true },
      Neuroticism: { type: Number, required: true }
    },
    riasec: {
      Realistic: { type: Number, required: true },
      Investigative: { type: Number, required: true },
      Artistic: { type: Number, required: true },
      Social: { type: Number, required: true },
      Enterprising: { type: Number, required: true },
      Conventional: { type: Number, required: true }
    }
  },
  derivedFeatures: {
    response_consistency: Number,
    response_time_variance: Number,
    straight_lining_score: Number,
    completion_rate: Number,
    average_response_time: Number
  },
  predictions: [{
    career: {
      type: String,
      required: true
    },
    probability: {
      type: Number,
      required: true
    }
  }],
  reportText: {
    summary: String,
    strengths: [String],
    development_areas: [String],
    action_steps: [String],
    resources: [{
      title: String,
      url: String
    }]
  },
  modelVersion: {
    type: String,
    required: true
  },
  deviceType: {
    type: String,
    enum: ['mobile', 'desktop'],
    required: true
  },
  durationSeconds: {
    type: Number,
    required: true
  },
  status: {
    type: String,
    enum: ['in_progress', 'completed', 'error'],
    default: 'in_progress'
  },
  ipfsHash: {
    type: String,
    default: null
  },
  blockchainHash: {
    type: String,
    default: null
  },
  reportHash: {
    type: String,
    default: null
  }
}, {
  timestamps: true
});

// Index for efficient querying
assessmentSchema.index({ userId: 1, createdAt: -1 });
assessmentSchema.index({ sessionId: 1 });
assessmentSchema.index({ status: 1 });

module.exports = mongoose.model('Assessment', assessmentSchema);