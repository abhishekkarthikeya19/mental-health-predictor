import React, { useState } from 'react';
import { exampleDataset, getDatasetStats, getModelPerformance } from '../utils/modelDemo';

const DatasetInfo = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const datasetStats = getDatasetStats();
  const modelPerformance = getModelPerformance();
  
  return (
    <div className="dataset-info">
      <h2>Dataset & Model Information</h2>
      
      <div className="info-tabs">
        <button 
          className={activeTab === 'overview' ? 'active' : ''} 
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={activeTab === 'examples' ? 'active' : ''} 
          onClick={() => setActiveTab('examples')}
        >
          Example Data
        </button>
        <button 
          className={activeTab === 'performance' ? 'active' : ''} 
          onClick={() => setActiveTab('performance')}
        >
          Model Performance
        </button>
      </div>
      
      <div className="info-content">
        {activeTab === 'overview' && (
          <div className="overview-section">
            <div className="stats-card">
              <h3>Dataset Statistics</h3>
              <div className="stat-item">
                <span className="stat-label">Total Samples:</span>
                <span className="stat-value">{datasetStats.totalSamples.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Training Samples:</span>
                <span className="stat-value">{datasetStats.trainingSamples.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Testing Samples:</span>
                <span className="stat-value">{datasetStats.testingSamples.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Normal Class:</span>
                <span className="stat-value">{datasetStats.classCounts.normal.toLocaleString()} samples</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Distressed Class:</span>
                <span className="stat-value">{datasetStats.classCounts.distressed.toLocaleString()} samples</span>
              </div>
            </div>
            
            <div className="stats-card">
              <h3>Data Sources</h3>
              <div className="source-list">
                {datasetStats.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <span className="source-name">{source.name}:</span>
                    <span className="source-count">{source.count.toLocaleString()} samples</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="stats-card">
              <h3>Most Frequent Words</h3>
              <div className="word-cloud">
                {datasetStats.topWords.map((word, index) => (
                  <div 
                    key={index} 
                    className="word-item"
                    style={{ 
                      fontSize: `${Math.max(0.8, word.count / 30)}rem`,
                      opacity: Math.max(0.6, word.count / 120)
                    }}
                  >
                    {word.text}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'examples' && (
          <div className="examples-section">
            <h3>Example Dataset Entries</h3>
            <p className="section-description">
              Below are examples from the training dataset used to train the mental health prediction model.
              The dataset contains text labeled as either normal (0) or showing signs of distress (1).
            </p>
            
            <div className="example-list">
              {exampleDataset.map((example, index) => (
                <div key={index} className={`example-item ${example.label === 1 ? 'distressed' : 'normal'}`}>
                  <div className="example-text">"{example.text}"</div>
                  <div className="example-meta">
                    <span className="example-label">
                      Label: <strong>{example.label === 1 ? 'Distressed' : 'Normal'}</strong>
                    </span>
                    <span className="example-source">Source: {example.source}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {activeTab === 'performance' && (
          <div className="performance-section">
            <div className="stats-card">
              <h3>Model Performance Metrics</h3>
              <div className="metric-item">
                <span className="metric-label">Accuracy:</span>
                <span className="metric-value">{(modelPerformance.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Precision:</span>
                <span className="metric-value">{(modelPerformance.precision * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Recall:</span>
                <span className="metric-value">{(modelPerformance.recall * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">F1 Score:</span>
                <span className="metric-value">{(modelPerformance.f1Score * 100).toFixed(1)}%</span>
              </div>
            </div>
            
            <div className="stats-card">
              <h3>Confusion Matrix</h3>
              <div className="confusion-matrix">
                <div className="matrix-row matrix-header">
                  <div className="matrix-cell"></div>
                  <div className="matrix-cell">Predicted Normal</div>
                  <div className="matrix-cell">Predicted Distressed</div>
                </div>
                <div className="matrix-row">
                  <div className="matrix-cell">Actual Normal</div>
                  <div className="matrix-cell matrix-value true-negative">
                    {modelPerformance.confusionMatrix.trueNegatives}
                  </div>
                  <div className="matrix-cell matrix-value false-positive">
                    {modelPerformance.confusionMatrix.falsePositives}
                  </div>
                </div>
                <div className="matrix-row">
                  <div className="matrix-cell">Actual Distressed</div>
                  <div className="matrix-cell matrix-value false-negative">
                    {modelPerformance.confusionMatrix.falseNegatives}
                  </div>
                  <div className="matrix-cell matrix-value true-positive">
                    {modelPerformance.confusionMatrix.truePositives}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="stats-card">
              <h3>Additional Information</h3>
              <div className="info-item">
                <span className="info-label">Training Time:</span>
                <span className="info-value">{modelPerformance.trainingTime}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Model Size:</span>
                <span className="info-value">{modelPerformance.modelSize}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Model Type:</span>
                <span className="info-value">DistilBERT (Transformer)</span>
              </div>
              <div className="info-item">
                <span className="info-label">Last Updated:</span>
                <span className="info-value">{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetInfo;