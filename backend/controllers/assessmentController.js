const Assessment = require('../models/Assessment');
const { calculateTraitScores, generateCareerReport } = require('../services/assessmentService');
const { storeReportOnChain } = require('../services/blockchain');
const { uploadToIPFS } = require('../services/ipfs');
const { encrypt, generateHash } = require('../services/encryption');
const axios = require('axios');

// Submit psychometric assessment
const submitAssessment = async (req, res) => {
  try {
    const { session_id, responses, duration_seconds, device_type, timestamp } = req.body;
    const userId = req.user._id;

    // Validate responses
    if (!responses || !Array.isArray(responses) || responses.length !== 35) {
      return res.status(400).json({ 
        message: 'Invalid responses. Expected 35 question responses.' 
      });
    }

    // Calculate trait scores from responses
    const traitScores = calculateTraitScores(responses);
    
    // Generate derived features
    const derivedFeatures = {
      response_consistency: calculateResponseConsistency(responses),
      response_time_variance: calculateResponseTimeVariance(responses),
      straight_lining_score: calculateStraightLining(responses),
      completion_rate: responses.length / 35,
      average_response_time: responses.reduce((sum, r) => sum + r.response_time_ms, 0) / responses.length
    };

    // Call Python ML service for predictions
    const modelResponse = await axios.post(
      `${process.env.MODEL_API_URL}/model/infer`,
      {
        trait_scores: traitScores,
        derived_features: derivedFeatures,
        session_metadata: {
          device_type,
          duration_seconds,
          timestamp
        }
      }
    );

    const predictions = modelResponse.data;

    // Generate human-readable report
    const reportText = await generateCareerReport(predictions, traitScores);

    // Create assessment record
    const assessment = new Assessment({
      userId,
      sessionId: session_id,
      responses,
      traitScores,
      derivedFeatures,
      predictions: predictions.predictions,
      reportText,
      modelVersion: predictions.model_version,
      deviceType: device_type,
      durationSeconds: duration_seconds,
      status: 'completed'
    });

    await assessment.save();

    res.status(201).json({
      message: 'Assessment completed successfully',
      assessment_id: assessment._id,
      predictions: predictions.predictions,
      trait_scores: traitScores,
      report_text: reportText,
      model_version: predictions.model_version
    });

  } catch (error) {
    console.error('Assessment submission error:', error);
    res.status(500).json({ 
      message: 'Failed to process assessment',
      error: error.message 
    });
  }
};

// Get assessment by ID
const getAssessment = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user._id;

    const assessment = await Assessment.findOne({ 
      _id: id, 
      userId 
    });

    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }

    res.json(assessment);

  } catch (error) {
    console.error('Get assessment error:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Get user's assessment history
const getAssessmentHistory = async (req, res) => {
  try {
    const userId = req.user._id;
    const { page = 1, limit = 10 } = req.query;

    const assessments = await Assessment.find({ userId })
      .sort({ createdAt: -1 })
      .limit(limit * 1)
      .skip((page - 1) * limit)
      .select('_id createdAt status predictions.0 modelVersion');

    const total = await Assessment.countDocuments({ userId });

    res.json({
      assessments,
      total,
      page: parseInt(page),
      pages: Math.ceil(total / limit)
    });

  } catch (error) {
    console.error('Get assessment history error:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Confirm report for blockchain storage
const confirmReportStorage = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user._id;
    const { store_on_blockchain = false } = req.body;

    const assessment = await Assessment.findOne({ 
      _id: id, 
      userId 
    });

    if (!assessment) {
      return res.status(404).json({ message: 'Assessment not found' });
    }

    if (assessment.ipfsHash) {
      return res.status(400).json({ message: 'Report already stored' });
    }

    if (store_on_blockchain) {
      // Encrypt report data
      const reportData = {
        predictions: assessment.predictions,
        trait_scores: assessment.traitScores,
        report_text: assessment.reportText,
        model_version: assessment.modelVersion,
        assessment_date: assessment.createdAt
      };

      const encryptedReport = encrypt(JSON.stringify(reportData));
      const reportHash = generateHash(JSON.stringify(reportData));

      // Upload to IPFS
      const ipfsCid = await uploadToIPFS(encryptedReport, `assessment_${assessment._id}.json`);

      // Store hash on blockchain
      const userAddress = `0x${userId.toString().padStart(40, '0')}`;
      const transactionHash = await storeReportOnChain(userAddress, ipfsCid, reportHash);

      // Update assessment record
      assessment.ipfsHash = ipfsCid;
      assessment.blockchainHash = transactionHash;
      assessment.reportHash = reportHash;
      await assessment.save();

      res.json({
        message: 'Report stored securely on blockchain and IPFS',
        ipfs_cid: ipfsCid,
        transaction_hash: transactionHash
      });
    } else {
      res.json({
        message: 'Report confirmed without blockchain storage'
      });
    }

  } catch (error) {
    console.error('Report storage error:', error);
    res.status(500).json({ 
      message: 'Failed to store report',
      error: error.message 
    });
  }
};

// Helper functions
const calculateResponseConsistency = (responses) => {
  // Calculate standard deviation of responses
  const values = responses.map(r => r.response);
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
};

const calculateResponseTimeVariance = (responses) => {
  const times = responses.map(r => r.response_time_ms);
  const mean = times.reduce((sum, time) => sum + time, 0) / times.length;
  const variance = times.reduce((sum, time) => sum + Math.pow(time - mean, 2), 0) / times.length;
  return Math.sqrt(variance);
};

const calculateStraightLining = (responses) => {
  // Detect if user gave same response to consecutive questions
  let straightLineCount = 0;
  for (let i = 1; i < responses.length; i++) {
    if (responses[i].response === responses[i-1].response) {
      straightLineCount++;
    }
  }
  return straightLineCount / (responses.length - 1);
};

module.exports = {
  submitAssessment,
  getAssessment,
  getAssessmentHistory,
  confirmReportStorage
};