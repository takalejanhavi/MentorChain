import React from 'react';
import { Download, FileText, Share2 } from 'lucide-react';

interface ReportDownloadProps {
  reportData: {
    predictions: Array<{
      career: string;
      probability: number;
    }>;
    trait_scores: {
      big_five: Record<string, number>;
      riasec: Record<string, number>;
    };
    report_text: {
      summary: string;
      strengths: string[];
      development_areas: string[];
      action_steps: string[];
      resources: Array<{
        title: string;
        url: string;
      }>;
    };
    model_version: string;
    assessment_date: string;
  };
  onDownloadPDF: () => void;
  onShare: () => void;
}

const ReportDownload: React.FC<ReportDownloadProps> = ({
  reportData,
  onDownloadPDF,
  onShare
}) => {
  const generateTextReport = () => {
    const { predictions, trait_scores, report_text, assessment_date } = reportData;
    
    let textReport = `CAREER GUIDANCE REPORT\n`;
    textReport += `Generated: ${new Date(assessment_date).toLocaleDateString()}\n`;
    textReport += `${'='.repeat(50)}\n\n`;
    
    textReport += `EXECUTIVE SUMMARY\n`;
    textReport += `${'-'.repeat(20)}\n`;
    textReport += `${report_text.summary}\n\n`;
    
    textReport += `TOP CAREER MATCHES\n`;
    textReport += `${'-'.repeat(20)}\n`;
    predictions.forEach((pred, index) => {
      textReport += `${index + 1}. ${pred.career} (${(pred.probability * 100).toFixed(1)}% match)\n`;
    });
    textReport += `\n`;
    
    textReport += `PERSONALITY PROFILE\n`;
    textReport += `${'-'.repeat(20)}\n`;
    textReport += `Big Five Traits:\n`;
    Object.entries(trait_scores.big_five).forEach(([trait, score]) => {
      textReport += `  ${trait}: ${score.toFixed(1)}/10\n`;
    });
    textReport += `\nRIASEC Interests:\n`;
    Object.entries(trait_scores.riasec).forEach(([interest, score]) => {
      textReport += `  ${interest}: ${score.toFixed(1)}/10\n`;
    });
    textReport += `\n`;
    
    textReport += `KEY STRENGTHS\n`;
    textReport += `${'-'.repeat(20)}\n`;
    report_text.strengths.forEach((strength, index) => {
      textReport += `${index + 1}. ${strength}\n`;
    });
    textReport += `\n`;
    
    textReport += `DEVELOPMENT AREAS\n`;
    textReport += `${'-'.repeat(20)}\n`;
    report_text.development_areas.forEach((area, index) => {
      textReport += `${index + 1}. ${area}\n`;
    });
    textReport += `\n`;
    
    textReport += `RECOMMENDED ACTIONS\n`;
    textReport += `${'-'.repeat(20)}\n`;
    report_text.action_steps.forEach((step, index) => {
      textReport += `${index + 1}. ${step}\n`;
    });
    textReport += `\n`;
    
    textReport += `LEARNING RESOURCES\n`;
    textReport += `${'-'.repeat(20)}\n`;
    report_text.resources.forEach((resource, index) => {
      textReport += `${index + 1}. ${resource.title}\n   ${resource.url}\n`;
    });
    
    return textReport;
  };

  const downloadTextReport = () => {
    const textContent = generateTextReport();
    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `career_report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
      <h3 className="text-xl font-semibold text-white mb-4">Download Your Report</h3>
      
      <div className="space-y-3">
        <button
          onClick={onDownloadPDF}
          className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg hover:shadow-lg transition-all duration-200"
        >
          <Download className="w-5 h-5" />
          <span>Download PDF Report</span>
        </button>
        
        <button
          onClick={downloadTextReport}
          className="w-full flex items-center justify-center space-x-2 bg-white/10 text-white py-3 px-4 rounded-lg hover:bg-white/20 transition-colors"
        >
          <FileText className="w-5 h-5" />
          <span>Download Text Report</span>
        </button>
        
        <button
          onClick={onShare}
          className="w-full flex items-center justify-center space-x-2 bg-white/10 text-white py-3 px-4 rounded-lg hover:bg-white/20 transition-colors"
        >
          <Share2 className="w-5 h-5" />
          <span>Share Report</span>
        </button>
      </div>
      
      <div className="mt-4 text-xs text-gray-400 text-center">
        Your report contains personalized career guidance based on validated psychometric assessment
      </div>
    </div>
  );
};

export default ReportDownload;