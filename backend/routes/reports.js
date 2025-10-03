const express = require('express');
const { authMiddleware, requireRole } = require('../middleware/auth');
const Report = require('../models/Report');
const { encrypt, decrypt, generateHash } = require('../services/encryption');
const { uploadToIPFS, retrieveFromIPFS } = require('../services/ipfs');
const { storeReportOnChain, verifyReportOnChain } = require('../services/blockchain');

const router = express.Router();

// Store encrypted report on IPFS and blockchain
router.post('/store-encrypted', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const reportData = req.body;
    const userId = req.user._id;
    const userAddress = req.user.walletAddress || `0x${userId.toString().padStart(40, '0')}`; // Generate address from user ID

    console.log('🔐 Encrypting and storing report...');

    // Encrypt the report
    const reportJson = JSON.stringify(reportData);
    const encryptedReport = encrypt(reportJson);
    const reportHash = generateHash(reportJson);

    // Upload to IPFS
    const cid = await uploadToIPFS(encryptedReport, `report-${userId}-${Date.now()}.json`);

    // Store on blockchain
    const transactionHash = await storeReportOnChain(userAddress, cid, reportHash);

    // Save metadata to database
    const reportRecord = new Report({
      userId,
      answers: reportData.analysis || {},
      report: {
        top3_careers: reportData.top3_career_fields || {},
        accuracy: reportData.accuracy || 95,
        recommendations: JSON.stringify(reportData.recommendations || {})
      },
      blockchainHash: transactionHash,
      ipfsCid: cid,
      reportHash: reportHash,
      status: 'completed'
    });

    await reportRecord.save();

    console.log('✅ Report stored successfully');

    res.json({
      success: true,
      cid,
      transaction_hash: transactionHash,
      report_id: reportRecord._id,
      message: 'Report encrypted and stored securely'
    });

  } catch (error) {
    console.error('Report storage error:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to store encrypted report',
      details: error.message 
    });
  }
});

// Retrieve and decrypt report from IPFS
router.get('/decrypt/:cid', authMiddleware, async (req, res) => {
  try {
    const { cid } = req.params;
    const userId = req.user._id;

    // Verify user owns this report
    const reportRecord = await Report.findOne({ userId, ipfsCid: cid });
    if (!reportRecord) {
      return res.status(403).json({ error: 'Access denied or report not found' });
    }

    // Retrieve from IPFS
    const encryptedReport = await retrieveFromIPFS(cid);

    // Decrypt the report
    const decryptedJson = decrypt(encryptedReport);
    const reportData = JSON.parse(decryptedJson);

    // Verify integrity
    const currentHash = generateHash(decryptedJson);
    const isValid = await verifyReportOnChain(cid, currentHash);

    res.json({
      success: true,
      report: reportData,
      integrity_verified: isValid,
      cid,
      transaction_hash: reportRecord.blockchainHash
    });

  } catch (error) {
    console.error('Report retrieval error:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve report',
      details: error.message 
    });
  }
});

// Verify report authenticity
router.post('/verify', authMiddleware, async (req, res) => {
  try {
    const { cid, reportHash } = req.body;

    if (!cid || !reportHash) {
      return res.status(400).json({ error: 'CID and report hash required' });
    }

    // Verify on blockchain
    const isValid = await verifyReportOnChain(cid, reportHash);

    // Additional verification: retrieve and check
    let ipfsAccessible = false;
    try {
      await retrieveFromIPFS(cid);
      ipfsAccessible = true;
    } catch (error) {
      console.log('IPFS verification failed:', error.message);
    }

    res.json({
      success: true,
      blockchain_verified: isValid,
      ipfs_accessible: ipfsAccessible,
      cid,
      verified_at: new Date().toISOString()
    });

  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ 
      error: 'Verification failed',
      details: error.message 
    });
  }
});

// Get user's reports
router.get('/', authMiddleware, async (req, res) => {
  try {
    const userId = req.user._id;
    
    const reports = await Report.find({ userId })
      .sort({ createdAt: -1 });

    res.json(reports);
  } catch (error) {
    console.error('Reports fetch error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get specific report
router.get('/:reportId', authMiddleware, async (req, res) => {
  try {
    const { reportId } = req.params;
    const userId = req.user._id;
    const userRole = req.user.role;

    let report;

    if (userRole === 'student') {
      // Students can only access their own reports
      report = await Report.findOne({ _id: reportId, userId });
    } else {
      // Psychologists need permission to access reports
      const Permission = require('../models/Permission');
      
      const permission = await Permission.findOne({
        reportId,
        psychologistId: userId,
        status: 'active',
        expiresAt: { $gt: new Date() }
      });

      if (!permission) {
        return res.status(403).json({ message: 'Access denied' });
      }

      report = await Report.findById(reportId);
      
      // Update access timestamp
      permission.accessedAt = new Date();
      await permission.save();
    }

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    res.json(report);
  } catch (error) {
    console.error('Report fetch error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;