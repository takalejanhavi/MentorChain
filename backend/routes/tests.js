const express = require('express'); 
const axios = require('axios');
const { authMiddleware, requireRole } = require('../middleware/auth');
const Report = require('../models/Report');

const router = express.Router();

// Submit test answers (student only)
router.post('/submit', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const { answers, report } = req.body;
    const userId = req.user._id;

    // Create report in database
    const newReport = new Report({
      userId,
      answers,
      report: {
        top3_careers: report.top3_careers,
        accuracy: 95,
        recommendations: 'AI-generated career recommendations based on your assessment results.'
      },
      status: 'completed'
    });

    // Store on blockchain (optional, continue even if fails)
    try {
      const blockchainResponse = await axios.post(
        process.env.BLOCKCHAIN_API_URL || 'http://localhost:5001/api/store',
        {
          reportId: newReport._id.toString(),
          userId: userId.toString(),
          reportData: {
            answers,
            predictions: report.top3_careers,
            timestamp: new Date().toISOString()
          }
        }
      );

      if (blockchainResponse.data.success) {
        newReport.blockchainHash = blockchainResponse.data.transactionHash;
      }
    } catch (blockchainError) {
      console.error('Blockchain storage error:', blockchainError.message);
    }

    await newReport.save();

    res.status(201).json({
      message: 'Test submitted successfully',
      reportId: newReport._id,
      report: newReport.report
    });
  } catch (error) {
    console.error('Test submission error:', error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get student's test history
router.get('/history', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const userId = req.user._id;
    
    const reports = await Report.find({ userId })
      .sort({ createdAt: -1 })
      .select('createdAt report.top3_careers status');

    res.json(reports);
  } catch (error) {
    console.error('Test history error:', error.message);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
