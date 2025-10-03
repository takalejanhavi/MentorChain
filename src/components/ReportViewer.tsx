import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Shield, CheckCircle, AlertCircle, ExternalLink } from 'lucide-react';
import { api } from '../services/api';

interface ReportViewerProps {
  reports: any[];
  onReportDecrypted?: (report: any) => void;
}

const ReportViewer: React.FC<ReportViewerProps> = ({ reports, onReportDecrypted }) => {
  const [loading, setLoading] = useState<string | null>(null);
  const [decryptedReports, setDecryptedReports] = useState<Record<string, any>>({});
  const [error, setError] = useState('');

  const handleDecryptReport = async (cid: string) => {
    setLoading(cid);
    setError('');

    try {
      const response = await api.get(`/reports/decrypt/${cid}`);
      
      setDecryptedReports(prev => ({
        ...prev,
        [cid]: response.data
      }));

      if (onReportDecrypted) {
        onReportDecrypted(response.data);
      }

    } catch (error: any) {
      console.error('Decryption error:', error);
      setError(error.response?.data?.details || 'Failed to decrypt report');
    }

    setLoading(null);
  };

  const handleVerifyReport = async (cid: string, reportHash: string) => {
    try {
      const response = await api.post('/reports/verify', { cid, reportHash });
      
      // Update the decrypted report with verification status
      setDecryptedReports(prev => ({
        ...prev,
        [cid]: {
          ...prev[cid],
          verification: response.data
        }
      }));

    } catch (error: any) {
      console.error('Verification error:', error);
      setError('Failed to verify report');
    }
  };

  if (reports.length === 0) {
    return (
      <div className="text-center py-8">
        <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">No Stored Reports</h3>
        <p className="text-gray-300">Your encrypted reports will appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-semibold text-white mb-4">Stored Reports</h3>
      
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 mb-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        </div>
      )}

      {reports.map((report) => {
        const isDecrypted = decryptedReports[report.ipfsCid];
        const isLoading = loading === report.ipfsCid;

        return (
          <motion.div
            key={report._id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10"
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-lg font-semibold text-white">
                  Career Report #{reports.indexOf(report) + 1}
                </h4>
                <p className="text-sm text-gray-300">
                  Created: {new Date(report.createdAt).toLocaleDateString()}
                </p>
              </div>
              
              <div className="flex items-center space-x-2">
                {report.ipfsCid && (
                  <a
                    href={`https://${report.ipfsCid}.ipfs.w3s.link/`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300"
                    title="View on IPFS"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                )}
                
                {report.blockchainHash && (
                  <a
                    href={`https://mumbai.polygonscan.com/tx/${report.blockchainHash}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:text-purple-300"
                    title="View on Polygon"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                )}
              </div>
            </div>

            {report.ipfsCid && (
              <div className="bg-white/5 rounded-lg p-3 mb-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-gray-400">IPFS CID</p>
                    <code className="text-xs text-blue-400">{report.ipfsCid}</code>
                  </div>
                  
                  {!isDecrypted && (
                    <button
                      onClick={() => handleDecryptReport(report.ipfsCid)}
                      disabled={isLoading}
                      className="flex items-center space-x-2 bg-blue-500/20 text-blue-400 px-3 py-1 rounded text-sm hover:bg-blue-500/30 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? (
                        <div className="w-3 h-3 border border-blue-400 border-top-transparent rounded-full animate-spin"></div>
                      ) : (
                        <Eye className="w-3 h-3" />
                      )}
                      <span>{isLoading ? 'Decrypting...' : 'Decrypt & View'}</span>
                    </button>
                  )}
                </div>
              </div>
            )}

            {isDecrypted && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-400">Report Decrypted Successfully</span>
                  </div>
                  
                  {report.reportHash && (
                    <button
                      onClick={() => handleVerifyReport(report.ipfsCid, report.reportHash)}
                      className="flex items-center space-x-1 text-xs text-purple-400 hover:text-purple-300"
                    >
                      <Shield className="w-3 h-3" />
                      <span>Verify Integrity</span>
                    </button>
                  )}
                </div>

                {isDecrypted.verification && (
                  <div className="bg-white/5 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Shield className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-white">Verification Results</span>
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-300">Blockchain Verified:</span>
                        <span className={isDecrypted.verification.blockchain_verified ? 'text-green-400' : 'text-red-400'}>
                          {isDecrypted.verification.blockchain_verified ? 'Valid' : 'Invalid'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">IPFS Accessible:</span>
                        <span className={isDecrypted.verification.ipfs_accessible ? 'text-green-400' : 'text-red-400'}>
                          {isDecrypted.verification.ipfs_accessible ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {isDecrypted.report && (
                  <div className="bg-white/5 rounded-lg p-4">
                    <h5 className="text-sm font-medium text-white mb-3">Career Predictions</h5>
                    <div className="space-y-2">
                      {Object.entries(isDecrypted.report.top3_career_fields || {}).map(([career, probability]: [string, any], index) => (
                        <div key={index} className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">{career}</span>
                          <span className="text-sm text-blue-400">{probability}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
};

export default ReportViewer;