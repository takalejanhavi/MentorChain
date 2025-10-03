import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Download, Calendar, Shield, TrendingUp } from 'lucide-react';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { api } from '../services/api';
import ReportViewer from '../components/ReportViewer';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface Report {
  _id: string;
  createdAt: string;
  ipfsCid?: string;
  report: {
    top3_careers: Record<string, number>;
  };
  answers: {
    Math_Score: number;
    Science_Score: number;
    English_Score: number;
    Extracurricular_Score: number;
    Openness: number;
    Conscientiousness: number;
    Extraversion: number;
    Agreeableness: number;
    Neuroticism: number;
    Percentage: number;
  };
}

const Reports: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [showEncrypted, setShowEncrypted] = useState(false);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const response = await api.get('/reports');
      setReports(response.data);
      if (response.data.length > 0) {
        setSelectedReport(response.data[0]);
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white text-xl">Loading reports...</div>
      </div>
    );
  }

  if (reports.length === 0) {
    return (
      <div className="text-center py-16">
        <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">No Reports Yet</h3>
        <p className="text-gray-300 mb-6">Take the career assessment to generate your first report</p>
        <button
          onClick={() => (window.location.href = '/take-test')}
          className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:shadow-lg transition-all duration-200"
        >
          Take Assessment
        </button>
      </div>
    );
  }

  const encryptedReports = reports.filter((report) => report.ipfsCid);
  const regularReports = reports.filter((report) => !report.ipfsCid);

  // --- Chart Data ---
  const careerMatchChart = selectedReport
    ? {
        labels: Object.keys(selectedReport.report.top3_careers),
        datasets: [
          {
            label: 'Career Match %',
            data: Object.values(selectedReport.report.top3_careers),
            backgroundColor: ['rgba(59, 130, 246, 0.8)', 'rgba(139, 92, 246, 0.8)', 'rgba(16, 185, 129, 0.8)'],
            borderColor: ['rgba(59, 130, 246, 1)', 'rgba(139, 92, 246, 1)', 'rgba(16, 185, 129, 1)'],
            borderWidth: 2,
          },
        ],
      }
    : null;

  const skillsChart = selectedReport
    ? {
        labels: ['Math', 'Science', 'English', 'Extracurricular', 'Personality Avg'],
        datasets: [
          {
            label: 'Your Scores',
            data: [
              selectedReport.answers.Math_Score,
              selectedReport.answers.Science_Score,
              selectedReport.answers.English_Score,
              selectedReport.answers.Extracurricular_Score * 10,
              ((selectedReport.answers.Openness +
                selectedReport.answers.Conscientiousness +
                selectedReport.answers.Extraversion +
                selectedReport.answers.Agreeableness +
                (11 - selectedReport.answers.Neuroticism)) /
                5) *
                10,
            ],
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderColor: 'rgba(59, 130, 246, 1)',
            pointBackgroundColor: 'rgba(59, 130, 246, 1)',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            fill: true,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: '#ffffff',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
      },
    },
    scales: {
      x: {
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
      },
      y: {
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
      },
    },
  };

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
          <h1 className="text-3xl font-bold text-white mb-2">Career Reports</h1>
          <p className="text-gray-300">View your comprehensive career guidance reports</p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowEncrypted(!showEncrypted)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              showEncrypted
                ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                : 'bg-white/10 text-white hover:bg-white/20'
            }`}
          >
            <Shield className="w-4 h-4" />
            <span>{showEncrypted ? 'Hide' : 'Show'} Encrypted Reports</span>
          </button>

          <button className="flex items-center space-x-2 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </motion.div>

      {/* Encrypted Reports */}
      {showEncrypted && encryptedReports.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20"
        >
          <div className="flex items-center space-x-2 mb-4">
            <Shield className="w-5 h-5 text-purple-400" />
            <h2 className="text-xl font-semibold text-white">Encrypted Reports (IPFS + Blockchain)</h2>
          </div>
          <ReportViewer reports={encryptedReports} />
        </motion.div>
      )}

      {/* Regular Reports */}
      {regularReports.length > 0 && selectedReport && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Reports List */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="lg:col-span-1"
          >
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">Report History</h3>
              <div className="space-y-3">
                {regularReports.map((report, index) => (
                  <button
                    key={report._id}
                    onClick={() => setSelectedReport(report)}
                    className={`w-full text-left p-4 rounded-lg transition-all duration-200 ${
                      selectedReport._id === report._id
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                        : 'bg-white/5 text-gray-300 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        <div
                          className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            selectedReport._id === report._id ? 'bg-white/20' : 'bg-white/10'
                          }`}
                        >
                          <BarChart3 className="w-5 h-5" />
                        </div>
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">Report #{index + 1}</p>
                        <div className="flex items-center space-x-1 text-sm opacity-75">
                          <Calendar className="w-3 h-3" />
                          <span>{new Date(report.createdAt).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Report Details */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="lg:col-span-3 space-y-6"
          >
            {/* Top Careers */}
            {careerMatchChart && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-xl font-semibold text-white mb-4">Top Career Matches</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Bar data={careerMatchChart} options={chartOptions} />
                  <div className="space-y-4">
                    {Object.entries(selectedReport.report.top3_careers).map(([career, probability], index) => (
                      <div key={index} className="bg-white/5 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold text-white">{career}</h4>
                          <span className="text-sm text-blue-400">#{index + 1}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                            style={{ width: `${probability}%` }}
                          ></div>
                        </div>
                        <p className="text-sm text-gray-300">{probability.toFixed(1)}% compatibility</p>
                        <p className="text-xs text-gray-400 mt-1">
                          {index === 0
                            ? 'Highest match - This career aligns best with your skills and personality'
                            : index === 1
                            ? 'Strong match - Consider exploring this career path further'
                            : 'Good match - Could be a viable alternative career option'}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Skills Analysis */}
            {skillsChart && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-xl font-semibold text-white mb-4">Skills & Personality Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Line data={skillsChart} options={chartOptions} />
                  <div className="space-y-4">
                    {/* Academic Strengths */}
                    <div className="bg-white/5 rounded-lg p-4">
                      <h4 className="font-semibold text-white mb-3">Academic Strengths</h4>
                      <div className="space-y-2">
                        {['Math', 'Science', 'English', 'Overall'].map((subject, i) => (
                          <div className="flex justify-between" key={i}>
                            <span className="text-gray-300">{subject}</span>
                            <span className="text-white font-medium">
                              {subject === 'Math'
                                ? selectedReport.answers.Math_Score + '%'
                                : subject === 'Science'
                                ? selectedReport.answers.Science_Score + '%'
                                : subject === 'English'
                                ? selectedReport.answers.English_Score + '%'
                                : selectedReport.answers.Percentage + '%'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Personality Traits */}
                    <div className="bg-white/5 rounded-lg p-4">
                      <h4 className="font-semibold text-white mb-3">Personality Traits</h4>
                      <div className="space-y-2">
                        {[
                          ['Openness', selectedReport.answers.Openness],
                          ['Conscientiousness', selectedReport.answers.Conscientiousness],
                          ['Extraversion', selectedReport.answers.Extraversion],
                          ['Agreeableness', selectedReport.answers.Agreeableness],
                          ['Emotional Stability', 11 - selectedReport.answers.Neuroticism],
                        ].map(([trait, value], i) => (
                          <div className="flex justify-between" key={i}>
                            <span className="text-gray-300">{trait}</span>
                            <span className="text-white font-medium">{value}/10</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Recommendations */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="text-xl font-semibold text-white mb-4">Career Recommendations</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {Object.entries(selectedReport.report.top3_careers).map(([career, probability], index) => (
                  <div key={index} className="bg-white/5 rounded-lg p-4">
                    <h4 className="font-semibold text-white mb-2">{career}</h4>
                    <p className="text-sm text-gray-300 mb-3">
                      {career === 'Doctor' &&
                        'Combine your strong science and math skills with helping others. Consider specializations in research or clinical practice.'}
                      {career === 'Engineer' &&
                        'Your analytical skills and problem-solving abilities make you well-suited for engineering fields.'}
                      {career === 'IT Professional' &&
                        'Technology careers offer diverse opportunities that match your logical thinking and innovation mindset.'}
                      {career === 'Designer' &&
                        'Creative fields where you can express your artistic vision while solving design challenges.'}
                      {career === 'Business/Entrepreneur' &&
                        'Leadership roles and business ventures that leverage your strategic thinking and communication skills.'}
                      {career === 'Teacher' &&
                        "Share your knowledge and make a difference in students' lives through education."}
                      {career === 'Psychologist' &&
                        'Help others while utilizing your understanding of human behavior and strong analytical skills.'}
                      {career === 'Scientist' &&
                        'Research and discovery in fields that match your curiosity and analytical approach.'}
                      {career === 'Lawyer' &&
                        'Legal profession where your analytical and communication skills can make a significant impact.'}
                      {career === 'Accountant' &&
                        'Financial and analytical work that requires attention to detail and mathematical skills.'}
                      {career === 'Artist' &&
                        'Creative expression through various artistic mediums and creative industries.'}
                      {career === 'Pilot' &&
                        'Aviation career combining technical skills with responsibility and precision.'}
                      {career === 'Musician' &&
                        'Musical career path leveraging your creative talents and artistic expression.'}
                    </p>
                    <div className="flex items-center text-xs text-blue-400">
                      <TrendingUp className="w-3 h-3 mr-1" />
                      <span>{probability.toFixed(1)}% match</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default Reports;
