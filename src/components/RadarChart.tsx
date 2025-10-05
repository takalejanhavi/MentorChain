import React from 'react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface RadarChartProps {
  data: {
    big_five: Record<string, number>;
    riasec: Record<string, number>;
  };
  type: 'big_five' | 'riasec';
  title: string;
}

const RadarChart: React.FC<RadarChartProps> = ({ data, type, title }) => {
  const chartData = type === 'big_five' ? data.big_five : data.riasec;
  
  const labels = type === 'big_five' 
    ? ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    : ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'];

  const values = labels.map(label => chartData[label] || 0);

  const radarData = {
    labels,
    datasets: [
      {
        label: 'Your Profile',
        data: values,
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.parsed.r.toFixed(1)}/10`;
          }
        }
      },
    },
    scales: {
      r: {
        beginAtZero: true,
        min: 0,
        max: 10,
        ticks: {
          stepSize: 2,
          color: '#9CA3AF',
          backdropColor: 'transparent',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        angleLines: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        pointLabels: {
          color: '#ffffff',
          font: {
            size: 12,
            weight: 'bold' as const,
          },
        },
      },
    },
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
      <h3 className="text-xl font-semibold text-white mb-4 text-center">{title}</h3>
      <div className="h-80">
        <Radar data={radarData} options={options} />
      </div>
      
      {/* Legend */}
      <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
        {labels.map((label, index) => (
          <div key={label} className="flex items-center justify-between text-gray-300">
            <span>{label}:</span>
            <span className="text-blue-400 font-medium">{values[index].toFixed(1)}/10</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RadarChart;