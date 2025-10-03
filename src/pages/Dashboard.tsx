import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Users, FileText, Shield, TrendingUp, Award } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../services/api';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalReports: 0,
    completedTests: 0,
    permissions: 0,
    accuracy: 0
  });
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await api.get('/dashboard');
      setStats(response.data.stats);
      setRecentActivity(response.data.recentActivity);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
    setLoading(false);
  };

  const statCards = [
    {
      title: 'Total Reports',
      value: stats.totalReports,
      icon: BarChart3,
      color: 'from-blue-500 to-blue-600',
      change: '+12%'
    },
    {
      title: 'Completed Tests',
      value: stats.completedTests,
      icon: FileText,
      color: 'from-purple-500 to-purple-600',
      change: '+8%'
    },
    {
      title: user?.role === 'student' ? 'Permissions Granted' : 'Students Accessed',
      value: stats.permissions,
      icon: user?.role === 'student' ? Shield : Users,
      color: 'from-green-500 to-green-600',
      change: '+15%'
    },
    {
      title: 'AI Accuracy',
      value: `${stats.accuracy}%`,
      icon: Award,
      color: 'from-orange-500 to-orange-600',
      change: '+2%'
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl p-8 border border-white/10"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              Welcome back, {user?.name}! 👋
            </h1>
            <p className="text-gray-300 text-lg">
              {user?.role === 'student' 
                ? 'Ready to discover your career path?' 
                : 'Help students find their perfect careers'}
            </p>
          </div>
          <div className="hidden md:block">
            <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white text-2xl font-bold">
                {user?.name?.charAt(0).toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((card, index) => {
          const Icon = card.icon;
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:bg-white/10 transition-all duration-200"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 bg-gradient-to-r ${card.color} rounded-lg`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <span className="text-green-400 text-sm font-medium">{card.change}</span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-1">{card.value}</h3>
              <p className="text-gray-400 text-sm">{card.title}</p>
            </motion.div>
          );
        })}
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10"
        >
          <h3 className="text-xl font-semibold text-white mb-6">Recent Activity</h3>
          <div className="space-y-4">
            {recentActivity.length === 0 ? (
              <p className="text-gray-400 text-center py-8">No recent activity</p>
            ) : (
              recentActivity.map((activity: any, index) => (
                <div key={index} className="flex items-center space-x-3 p-3 bg-white/5 rounded-lg">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <div>
                    <p className="text-white text-sm">{activity.description}</p>
                    <p className="text-gray-400 text-xs">{activity.timestamp}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10"
        >
          <h3 className="text-xl font-semibold text-white mb-6">Quick Actions</h3>
          <div className="space-y-3">
            {user?.role === 'student' ? (
              <>
                <button className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 rounded-lg hover:shadow-lg transition-all duration-200 text-left">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-5 h-5" />
                    <div>
                      <p className="font-medium">Take Career Assessment</p>
                      <p className="text-sm opacity-75">Discover your ideal career path</p>
                    </div>
                  </div>
                </button>
                <button className="w-full bg-white/10 text-white p-4 rounded-lg hover:bg-white/20 transition-all duration-200 text-left">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-5 h-5" />
                    <div>
                      <p className="font-medium">View Reports</p>
                      <p className="text-sm opacity-75">Check your career guidance reports</p>
                    </div>
                  </div>
                </button>
                <button className="w-full bg-white/10 text-white p-4 rounded-lg hover:bg-white/20 transition-all duration-200 text-left">
                  <div className="flex items-center space-x-3">
                    <Shield className="w-5 h-5" />
                    <div>
                      <p className="font-medium">Manage Permissions</p>
                      <p className="text-sm opacity-75">Control who can access your reports</p>
                    </div>
                  </div>
                </button>
              </>
            ) : (
              <>
                <button className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 rounded-lg hover:shadow-lg transition-all duration-200 text-left">
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5" />
                    <div>
                      <p className="font-medium">Access Student Reports</p>
                      <p className="text-sm opacity-75">View permissioned student reports</p>
                    </div>
                  </div>
                </button>
                <button className="w-full bg-white/10 text-white p-4 rounded-lg hover:bg-white/20 transition-all duration-200 text-left">
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="w-5 h-5" />
                    <div>
                      <p className="font-medium">Analytics Dashboard</p>
                      <p className="text-sm opacity-75">View student progress analytics</p>
                    </div>
                  </div>
                </button>
              </>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;