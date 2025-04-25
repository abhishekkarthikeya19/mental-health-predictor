import React, { useState, useEffect } from 'react';
import { Line, Bar, Pie } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';

// Register Chart.js components
Chart.register(...registerables);

const Dashboard = ({ analysisData }) => {
  const [activeTab, setActiveTab] = useState('sentiment');
  
  if (!analysisData) {
    return (
      <div className="dashboard-container">
        <div className="dashboard-message">
          <p>No analysis data available. Submit text for analysis to see insights.</p>
        </div>
      </div>
    );
  }
  
  const { 
    sentimentTrends, 
    topWords, 
    emotionDistribution, 
    languagePatterns,
    userActivity,
    predictionHistory
  } = analysisData;
  
  // Format sentiment trend data for Chart.js
  const sentimentChartData = {
    labels: sentimentTrends?.dates || [],
    datasets: [
      {
        label: 'Positive Sentiment',
        data: sentimentTrends?.positive || [],
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Negative Sentiment',
        data: sentimentTrends?.negative || [],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: true,
        tension: 0.4
      }
    ]
  };
  
  // Format word frequency data for Chart.js
  const wordFrequencyData = {
    labels: topWords?.map(word => word.text) || [],
    datasets: [
      {
        label: 'Word Frequency',
        data: topWords?.map(word => word.count) || [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  // Format emotion distribution data for Chart.js
  const emotionChartData = {
    labels: emotionDistribution?.map(item => item.emotion) || [],
    datasets: [
      {
        data: emotionDistribution?.map(item => item.percentage) || [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)'
        ],
        borderWidth: 1
      }
    ]
  };
  
  // Format language patterns data for Chart.js
  const languagePatternsData = {
    labels: languagePatterns?.categories || [],
    datasets: [
      {
        label: 'Normal',
        data: languagePatterns?.normal || [],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderWidth: 1
      },
      {
        label: 'Distressed',
        data: languagePatterns?.distressed || [],
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderWidth: 1
      }
    ]
  };
  
  // Format user activity data for Chart.js
  const userActivityData = {
    labels: userActivity?.dates || [],
    datasets: [
      {
        label: 'Activity Level',
        data: userActivity?.counts || [],
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        fill: true,
        tension: 0.4
      }
    ]
  };
  
  // Format prediction history data for Chart.js
  const predictionHistoryData = {
    labels: predictionHistory?.dates || [],
    datasets: [
      {
        label: 'Prediction Confidence',
        data: predictionHistory?.confidence || [],
        borderColor: 'rgba(255, 159, 64, 1)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        fill: true,
        tension: 0.4
      }
    ]
  };
  
  return (
    <div className="dashboard-container">
      <div className="dashboard-tabs">
        <button 
          className={activeTab === 'sentiment' ? 'active' : ''} 
          onClick={() => setActiveTab('sentiment')}
        >
          Sentiment Trends
        </button>
        <button 
          className={activeTab === 'words' ? 'active' : ''} 
          onClick={() => setActiveTab('words')}
        >
          Word Frequency
        </button>
        <button 
          className={activeTab === 'emotions' ? 'active' : ''} 
          onClick={() => setActiveTab('emotions')}
        >
          Emotion Distribution
        </button>
        <button 
          className={activeTab === 'language' ? 'active' : ''} 
          onClick={() => setActiveTab('language')}
        >
          Language Patterns
        </button>
        <button 
          className={activeTab === 'activity' ? 'active' : ''} 
          onClick={() => setActiveTab('activity')}
        >
          User Activity
        </button>
        <button 
          className={activeTab === 'predictions' ? 'active' : ''} 
          onClick={() => setActiveTab('predictions')}
        >
          Prediction History
        </button>
      </div>
      
      <div className="dashboard-content">
        {activeTab === 'sentiment' && (
          <div className="chart-container">
            <h3>Sentiment Trends Over Time</h3>
            {sentimentTrends?.dates?.length > 0 ? (
              <>
                <Line 
                  data={sentimentChartData} 
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 1
                      }
                    },
                    plugins: {
                      title: {
                        display: true,
                        text: 'Sentiment Analysis Over Time'
                      },
                      tooltip: {
                        callbacks: {
                          label: function(context) {
                            return `${context.dataset.label}: ${(context.raw * 100).toFixed(1)}%`;
                          }
                        }
                      }
                    }
                  }}
                />
                <div className="chart-description">
                  <p>This chart shows the trends in positive and negative sentiment over time. Higher values indicate stronger sentiment intensity.</p>
                  <p>Monitoring these trends can help identify periods of emotional distress or improvement.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display sentiment trends.</p>
            )}
          </div>
        )}
        
        {activeTab === 'words' && (
          <div className="chart-container">
            <h3>Most Frequent Words</h3>
            {topWords?.length > 0 ? (
              <>
                <Bar 
                  data={wordFrequencyData} 
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        display: false
                      },
                      title: {
                        display: true,
                        text: 'Most Frequent Words in Text'
                      }
                    }
                  }}
                />
                <div className="chart-description">
                  <p>This chart displays the most frequently used words in the analyzed text.</p>
                  <p>The vocabulary and word choice can provide insights into mental state and emotional themes.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display word frequencies.</p>
            )}
          </div>
        )}
        
        {activeTab === 'emotions' && (
          <div className="chart-container">
            <h3>Emotion Distribution</h3>
            {emotionDistribution?.length > 0 ? (
              <>
                <div className="pie-chart-container">
                  <Pie 
                    data={emotionChartData} 
                    options={{
                      responsive: true,
                      plugins: {
                        title: {
                          display: true,
                          text: 'Distribution of Emotions'
                        },
                        tooltip: {
                          callbacks: {
                            label: function(context) {
                              return `${context.label}: ${context.raw.toFixed(1)}%`;
                            }
                          }
                        }
                      }
                    }}
                  />
                </div>
                <div className="chart-description">
                  <p>This chart shows the distribution of different emotions detected in the text.</p>
                  <p>A balanced emotional profile typically indicates better mental health, while dominance of negative emotions may suggest distress.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display emotion distribution.</p>
            )}
          </div>
        )}
        
        {activeTab === 'language' && (
          <div className="chart-container">
            <h3>Language Pattern Comparison</h3>
            {languagePatterns?.categories?.length > 0 ? (
              <>
                <Bar 
                  data={languagePatternsData} 
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true
                      }
                    },
                    plugins: {
                      title: {
                        display: true,
                        text: 'Language Patterns: Normal vs. Distressed'
                      }
                    }
                  }}
                />
                <div className="chart-description">
                  <p>This chart compares language patterns between normal and distressed states.</p>
                  <p>Significant differences in these patterns can help identify shifts in mental health status.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display language patterns.</p>
            )}
          </div>
        )}
        
        {activeTab === 'activity' && (
          <div className="chart-container">
            <h3>User Activity Over Time</h3>
            {userActivity?.dates?.length > 0 ? (
              <>
                <Line 
                  data={userActivityData} 
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true
                      }
                    },
                    plugins: {
                      title: {
                        display: true,
                        text: 'User Activity Levels'
                      }
                    }
                  }}
                />
                <div className="chart-description">
                  <p>This chart tracks user activity levels over time.</p>
                  <p>Sudden changes in activity patterns can sometimes indicate shifts in mental health status.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display user activity.</p>
            )}
          </div>
        )}
        
        {activeTab === 'predictions' && (
          <div className="chart-container">
            <h3>Prediction History</h3>
            {predictionHistory?.dates?.length > 0 ? (
              <>
                <Line 
                  data={predictionHistoryData} 
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 1
                      }
                    },
                    plugins: {
                      title: {
                        display: true,
                        text: 'Mental Health Prediction Confidence'
                      },
                      tooltip: {
                        callbacks: {
                          label: function(context) {
                            return `Confidence: ${(context.raw * 100).toFixed(1)}%`;
                          }
                        }
                      }
                    }
                  }}
                />
                <div className="chart-description">
                  <p>This chart shows the confidence levels of mental health predictions over time.</p>
                  <p>Higher values indicate stronger confidence in the detected mental health status.</p>
                </div>
              </>
            ) : (
              <p className="no-data-message">Not enough data to display prediction history.</p>
            )}
          </div>
        )}
      </div>
      
      <div className="dashboard-insights">
        <h3>Key Insights</h3>
        <ul>
          {analysisData.insights?.map((insight, index) => (
            <li key={index} className={insight.type}>
              <span className="insight-icon">{insight.type === 'warning' ? '⚠️' : '💡'}</span>
              <span className="insight-text">{insight.text}</span>
            </li>
          ))}
          {(!analysisData.insights || analysisData.insights.length === 0) && (
            <li>No specific insights available. Continue providing data for more detailed analysis.</li>
          )}
        </ul>
      </div>
      
      <div className="dashboard-footer">
        <p className="disclaimer">
          <strong>Note:</strong> This dashboard is for informational purposes only and should not be used as a substitute for professional mental health advice.
        </p>
      </div>
    </div>
  );
};

export default Dashboard;