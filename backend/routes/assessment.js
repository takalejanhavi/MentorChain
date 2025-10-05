const express = require('express');
const { authMiddleware, requireRole } = require('../middleware/auth');
const {
  submitAssessment,
  getAssessment,
  getAssessmentHistory,
  confirmReportStorage
} = require('../controllers/assessmentController');

const router = express.Router();

// Submit psychometric assessment
router.post('/submit', authMiddleware, submitAssessment);

// Get specific assessment
router.get('/:id', authMiddleware, getAssessment);

// Get assessment history
router.get('/', authMiddleware, getAssessmentHistory);

// Confirm report for blockchain storage
router.post('/:id/confirm', authMiddleware, confirmReportStorage);

module.exports = router;