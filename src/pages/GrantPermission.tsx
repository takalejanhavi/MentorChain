import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, Users, Mail, Check, X, Plus } from 'lucide-react';
import { api } from '../services/api';

const GrantPermission = () => {
  const [permissions, setPermissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newPermission, setNewPermission] = useState({
    psychologistEmail: '',
    reportId: '',
    duration: '30' // days
  });
  const [reports, setReports] = useState<any[]>([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [permissionsRes, reportsRes] = await Promise.all([
        api.get('/permissions'),
        api.get('/reports')
      ]);
      
      setPermissions(permissionsRes.data);
      setReports(reportsRes.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
    setLoading(false);
  };

  const handleGrantPermission = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      await api.post('/permissions/grant', {
        psychologistEmail: newPermission.psychologistEmail,
        reportId: newPermission.reportId,
        duration: parseInt(newPermission.duration)
      });
      
      setShowAddModal(false);
      setNewPermission({ psychologistEmail: '', reportId: '', duration: '30' });
      fetchData();
    } catch (error) {
      console.error('Error granting permission:', error);
    }
  };

  const handleRevokePermission = async (permissionId: string) => {
    try {
      await api.delete(`/permissions/${permissionId}`);
      fetchData();
    } catch (error) {
      console.error('Error revoking permission:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">Loading permissions...</div>
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
          <h1 className="text-3xl font-bold text-white mb-2">Permission Management</h1>
          <p className="text-gray-300">Control who can access your career reports</p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg hover:shadow-lg transition-all duration-200"
        >
          <Plus className="w-5 h-5" />
          <span>Grant Permission</span>
        </button>
      </motion.div>

      {/* Permissions List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10"
      >
        {permissions.length === 0 ? (
          <div className="text-center py-16">
            <Shield className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Permissions Granted</h3>
            <p className="text-gray-300 mb-6">You haven't granted access to any psychologists yet</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:shadow-lg transition-all duration-200"
            >
              Grant First Permission
            </button>
          </div>
        ) : (
          <div className="divide-y divide-white/10">
            {permissions.map((permission, index) => (
              <div key={permission._id} className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                      <Users className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {permission.psychologistId?.name || 'Unknown Psychologist'}
                      </h3>
                      <p className="text-gray-300 text-sm flex items-center space-x-1">
                        <Mail className="w-4 h-4" />
                        <span>{permission.psychologistId?.email || 'N/A'}</span>
                      </p>
                      <p className="text-gray-400 text-xs mt-1">
                        Report #{reports.findIndex(r => r._id === permission.reportId) + 1} • 
                        Granted {new Date(permission.createdAt).toLocaleDateString()} • 
                        Expires {new Date(permission.expiresAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className={`flex items-center space-x-1 px-3 py-1 rounded-full text-sm ${
                      permission.status === 'active' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {permission.status === 'active' ? (
                        <Check className="w-4 h-4" />
                      ) : (
                        <X className="w-4 h-4" />
                      )}
                      <span className="capitalize">{permission.status}</span>
                    </div>
                    
                    {permission.status === 'active' && (
                      <button
                        onClick={() => handleRevokePermission(permission._id)}
                        className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
                      >
                        Revoke
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </motion.div>

      {/* Grant Permission Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="bg-slate-800 rounded-2xl p-8 max-w-md w-full mx-4 border border-white/10"
          >
            <h3 className="text-2xl font-bold text-white mb-6">Grant Permission</h3>
            
            <form onSubmit={handleGrantPermission} className="space-y-6">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Psychologist Email
                </label>
                <input
                  type="email"
                  value={newPermission.psychologistEmail}
                  onChange={(e) => setNewPermission(prev => ({ ...prev, psychologistEmail: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  placeholder="Enter psychologist's email"
                  required
                />
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Select Report
                </label>
                <select
                  value={newPermission.reportId}
                  onChange={(e) => setNewPermission(prev => ({ ...prev, reportId: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  required
                >
                  <option value="" className="bg-slate-800">Select a report</option>
                  {reports.map((report, index) => (
                    <option key={report._id} value={report._id} className="bg-slate-800">
                      Report #{index + 1} - {new Date(report.createdAt).toLocaleDateString()}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Access Duration (days)
                </label>
                <select
                  value={newPermission.duration}
                  onChange={(e) => setNewPermission(prev => ({ ...prev, duration: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="7" className="bg-slate-800">7 days</option>
                  <option value="30" className="bg-slate-800">30 days</option>
                  <option value="60" className="bg-slate-800">60 days</option>
                  <option value="90" className="bg-slate-800">90 days</option>
                </select>
              </div>

              <div className="flex space-x-4">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-4 py-3 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all duration-200"
                >
                  Grant Access
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default GrantPermission;