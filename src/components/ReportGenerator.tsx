import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Shield, ExternalLink, CheckCircle, AlertCircle } from 'lucide-react';
import { api } from '../services/api';

interface ReportGeneratorProps {
  reportData: any;
  onReportStored?: (result: any) => void;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({ reportData, onReportStored }) => {
  const [storing, setStoring] = useState(false);
  const [stored, setStored] = useState(false);
  const [storageResult, setStorageResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleStoreReport = async () => {
    setStoring(true);
    setError('');

    try {
      const response = await api.post('/reports/store-encrypted', reportData);
      
      setStorageResult(response.data);
      setStored(true);
      
      if (onReportStored) {
        onReportStored(response.data);
      }

    } catch (error: any) {
      console.error('Storage error:', error);
      setError(error.response?.data?.details || 'Failed to store report securely');
    }

    setStoring(false);
  };

  if (stored && storageResult) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10"
      >
        <div className="flex items-center space-x-3 mb-4">
          <CheckCircle className="w-6 h-6 text-green-400" />
          <h3 className="text-lg font-semibold text-white">Report Stored Securely</h3>
        </div>

        <div className="space-y-4">
          <div className="bg-white/5 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-300 mb-2">IPFS Storage</h4>
            <div className="flex items-center space-x-2">
              <code className="text-xs bg-black/20 px-2 py-1 rounded text-blue-400 flex-1">
                {storageResult.cid}
              </code>
              <a
                href={`https://${storageResult.cid}.ipfs.w3s.link/`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>

          <div className="bg-white/5 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Blockchain Transaction</h4>
            <div className="flex items-center space-x-2">
              <code className="text-xs bg-black/20 px-2 py-1 rounded text-purple-400 flex-1">
                {storageResult.transaction_hash}
              </code>
              <a
                href={`https://mumbai.polygonscan.com/tx/${storageResult.transaction_hash}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-sm text-green-400">
            <Shield className="w-4 h-4" />
            <span>Report encrypted and stored immutably on IPFS + Polygon</span>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10"
    >
      <div className="text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
          <Upload className="w-8 h-8 text-white" />
        </div>

        <h3 className="text-xl font-semibold text-white mb-2">Secure Report Storage</h3>
        <p className="text-gray-300 mb-6">
          Store your career report securely on IPFS with blockchain verification
        </p>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 mb-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          </div>
        )}

        <button
          onClick={handleStoreReport}
          disabled={storing}
          className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 mx-auto"
        >
          {storing ? (
            <>
              <div className="w-4 h-4 border-2 border-white/20 border-top-white rounded-full animate-spin"></div>
              <span>Storing Securely...</span>
            </>
          ) : (
            <>
              <Shield className="w-4 h-4" />
              <span>Store on Blockchain + IPFS</span>
            </>
          )}
        </button>

        <div className="mt-4 text-xs text-gray-400 space-y-1">
          <p>• Report will be AES-256 encrypted</p>
          <p>• Stored on IPFS via Web3.Storage</p>
          <p>• Hash recorded on Polygon Mumbai</p>
          <p>• Immutable and verifiable</p>
        </div>
      </div>
    </motion.div>
  );
};

export default ReportGenerator;