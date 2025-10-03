const express = require('express');
const { authMiddleware, requireRole } = require('../middleware/auth');
const Permission = require('../models/Permission');
const User = require('../models/User');
const Report = require('../models/Report');

const router = express.Router();

/**
 * Grant permission to a psychologist to access a student's report
 * POST /api/permissions/grant
 * Access: Student only
 */
router.post('/grant', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const { psychologistEmail, reportId, duration } = req.body;
    const studentId = req.user._id;

    // Find psychologist
    const psychologist = await User.findOne({ email: psychologistEmail, role: 'psychologist', isActive: true });
    if (!psychologist) return res.status(404).json({ message: 'Psychologist not found' });

    // Verify report belongs to student
    const report = await Report.findOne({ _id: reportId, userId: studentId });
    if (!report) return res.status(404).json({ message: 'Report not found' });

    // Check for existing active permission
    const existingPermission = await Permission.findOne({
      studentId,
      psychologistId: psychologist._id,
      reportId,
      status: 'active'
    });
    if (existingPermission) return res.status(400).json({ message: 'Permission already granted' });

    // Calculate expiration
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + parseInt(duration));

    // Save permission
    const permission = await Permission.create({
      studentId,
      psychologistId: psychologist._id,
      reportId,
      expiresAt,
      status: 'active'
    });

    res.status(201).json({
      message: 'Permission granted successfully',
      permission: {
        ...permission.toObject(),
        psychologist: {
          name: psychologist.name,
          email: psychologist.email
        }
      }
    });

  } catch (error) {
    console.error('Grant permission error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * Get all permissions for a student
 * GET /api/permissions
 * Access: Student only
 */
router.get('/', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const studentId = req.user._id;
    const permissions = await Permission.find({ studentId })
      .populate('psychologistId', 'name email')
      .populate('reportId', 'createdAt top3_careers reportHash') // include blockchain hash
      .sort({ createdAt: -1 });

    res.json(permissions);
  } catch (error) {
    console.error('Permissions fetch error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * Get all reports a psychologist has access to
 * GET /api/permissions/accessible-reports
 * Access: Psychologist only
 */
router.get('/accessible-reports', authMiddleware, requireRole(['psychologist']), async (req, res) => {
  try {
    const psychologistId = req.user._id;

    const permissions = await Permission.find({
      psychologistId,
      status: 'active',
      expiresAt: { $gt: new Date() }
    })
      .populate('studentId', 'name email')
      .populate('reportId', 'createdAt top3_careers reportHash')
      .sort({ createdAt: -1 });

    res.json(permissions);
  } catch (error) {
    console.error('Accessible reports fetch error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * Revoke a permission
 * DELETE /api/permissions/:permissionId
 * Access: Student only
 */
router.delete('/:permissionId', authMiddleware, requireRole(['student']), async (req, res) => {
  try {
    const { permissionId } = req.params;
    const studentId = req.user._id;

    const permission = await Permission.findOne({ _id: permissionId, studentId });
    if (!permission) return res.status(404).json({ message: 'Permission not found' });

    permission.status = 'revoked';
    await permission.save();

    res.json({ message: 'Permission revoked successfully' });
  } catch (error) {
    console.error('Revoke permission error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
