import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Users, Search, Eye, BarChart3, Calendar, Shield } from 'lucide-react';
import { api } from '../services/api';

const AccessReports = () => {
  const [accessibleReports, setAccessibleReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [showReportModal, setShowReportModal] = useState(false);

  useEffect(() => {
    fetchAccessibleReports();
  }, []);

  const fetchAccessibleReports = async () => {
    try {
      const response = await api.get('/permissions/accessible-reports');
      setAccessibleReports(response.data);
    } catch (error) {
      console.error('Error fetching accessible reports:', error);
    }
    setLoading(false);
  };

  const handleViewReport = async (permission: any) => {
    try {
      const response = await api.get(`/reports/${permission.reportId._id}`);
      setSelectedReport({
        ...response.data,
        studentInfo: permission.studentId,
        permission: permission
      });
      setShowReportModal(true);
    } catch (error) {
      console.error('Error fetching report details:', error);
    }
  };

  const filteredReports = accessibleReports.filter(permission =>
    permission.studentId?.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    permission.studentId?.email?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">Loading accessible reports...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Student Reports Access</h1>
          <p className="text-gray-300">View reports that students have granted you access to</p>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search students..."
            className="pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </motion.div>

      {/* Reports Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6"
      >
        {filteredReports.length === 0 ? (
          <div className="col-span-full text-center py-16">
            <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Accessible Reports</h3>
            <p className="text-gray-300">Students haven't granted you access to any reports yet</p>
          </div>
        ) : (
          filteredReports.map((permission) => (
            <div
              key={permission._id}
              className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:bg-white/10 transition-all duration-200"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-semibold">
                      {permission.studentId?.name?.charAt(0).toUpperCase() || 'S'}
                    </span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">
                      {permission.studentId?.name || 'Unknown Student'}
                    </h3>
                    <p className="text-sm text-gray-300">{permission.studentId?.email || 'N/A'}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-1 px-2 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">
                  <Shield className="w-3 h-3" />
                  <span>Active</span>
                </div>
              </div>

              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">Report Date:</span>
                  <span className="text-white">
                    {new Date(permission.reportId?.createdAt).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">Access Granted:</span>
                  <span className="text-white">
                    {new Date(permission.createdAt).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">Expires:</span>
                  <span className="text-white">
                    {new Date(permission.expiresAt).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => handleViewReport(permission)}
                  className="flex-1 flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white py-2 px-4 rounded-lg hover:shadow-lg transition-all duration-200"
                >
                  <Eye className="w-4 h-4" />
                  <span>View Report</span>
                </button>
              </div>
            </div>
          ))
        )}
      </motion.div>

      {/* Report Modal */}
      {showReportModal && selectedReport && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="bg-slate-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-white/10"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div>
                <h3 className="text-2xl font-bold text-white">
                  {selectedReport.studentInfo?.name}'s Career Report
                </h3>
                <p className="text-gray-300 text-sm">
                  Generated on {new Date(selectedReport.createdAt).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={() => setShowReportModal(false)}
                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              {/* Top Careers */}
              <div>
                <h4 className="text-xl font-semibold text-white mb-4">Top Career Matches</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(selectedReport.report.top3_careers).map(([career, probability]: [string, any], index) => (
                    <div key={index} className="bg-white/5 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-semibold text-white">{career}</h5>
                        <span className="text-sm text-blue-400">#{index + 1}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                          style={{ width: `${probability}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-gray-300">{probability.toFixed(1)}% match</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Academic Scores */}
              <div>
                <h4 className="text-xl font-semibold text-white mb-4">Academic Performance</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white/5 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {selectedReport.answers.Math_Score}%
                    </div>
                    <div className="text-sm text-gray-300">Mathematics</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {selectedReport.answers.Science_Score}%
                    </div>
                    <div className="text-sm text-gray-300">Science</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {selectedReport.answers.English_Score}%
                    </div>
                    <div className="text-sm text-gray-300">English</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {selectedReport.answers.Percentage}%
                    </div>
                    <div className="text-sm text-gray-300">Overall</div>
                  </div>
                </div>
              </div>

              {/* Personality Traits */}
              <div>
                <h4 className="text-xl font-semibold text-white mb-4">Personality Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    {[
                      { label: 'Openness', value: selectedReport.answers.Openness },
                      { label: 'Conscientiousness', value: selectedReport.answers.Conscientiousness },
                      { label: 'Extraversion', value: selectedReport.answers.Extraversion }
                    ].map((trait, index) => (
                      <div key={index} className="bg-white/5 rounded-lg p-3">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-white font-medium">{trait.label}</span>
                          <span className="text-blue-400">{trait.value}/10</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                            style={{ width: `${(trait.value / 10) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-3">
                    {[
                      { label: 'Agreeableness', value: selectedReport.answers.Agreeableness },
                      { label: 'Emotional Stability', value: 11 - selectedReport.answers.Neuroticism },
                      { label: 'Extracurricular', value: selectedReport.answers.Extracurricular_Score }
                    ].map((trait, index) => (
                      <div key={index} className="bg-white/5 rounded-lg p-3">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-white font-medium">{trait.label}</span>
                          <span className="text-blue-400">{trait.value}/10</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                            style={{ width: `${(trait.value / 10) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default AccessReports;