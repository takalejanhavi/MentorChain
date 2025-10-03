const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const Report = require('../models/Report');
const Permission = require('../models/Permission');

const router = express.Router();

// Get dashboard stats
router.get('/', authMiddleware, async (req, res) => {
  try {
    const userId = req.user._id;
    const userRole = req.user.role;

    let stats = {
      totalReports: 0,
      completedTests: 0,
      permissions: 0,
      accuracy: 95
    };

    let recentActivity = [];

    if (userRole === 'student') {
      // Student stats
      const reports = await Report.find({ userId }).sort({ createdAt: -1 });
      const permissions = await Permission.find({ studentId: userId, status: 'active' });

      stats.totalReports = reports.length;
      stats.completedTests = reports.length;
      stats.permissions = permissions.length;

      // Recent activity for students
      recentActivity = reports.slice(0, 5).map(report => ({
        description: `Career assessment completed`,
        timestamp: report.createdAt.toLocaleDateString()
      }));

      permissions.slice(0, 3).forEach(permission => {
        recentActivity.push({
          description: `Granted report access to psychologist`,
          timestamp: permission.createdAt.toLocaleDateString()
        });
      });
    } else {
      // Psychologist stats
      const accessibleReports = await Permission.find({ 
        psychologistId: userId, 
        status: 'active' 
      }).populate('reportId');

      const uniqueStudents = new Set(
        accessibleReports.map(p => p.studentId.toString())
      ).size;

      stats.totalReports = accessibleReports.length;
      stats.completedTests = accessibleReports.length;
      stats.permissions = uniqueStudents;

      // Recent activity for psychologists
      recentActivity = accessibleReports.slice(0, 5).map(permission => ({
        description: `Gained access to student report`,
        timestamp: permission.createdAt.toLocaleDateString()
      }));
    }

    // Sort activity by most recent
    recentActivity.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    res.json({
      stats,
      recentActivity: recentActivity.slice(0, 10)
    });
  } catch (error) {
    console.error('Dashboard error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;