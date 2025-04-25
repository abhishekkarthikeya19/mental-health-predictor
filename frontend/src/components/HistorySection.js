import React, { useState, useEffect } from 'react';

const HistorySection = ({ onSelectItem }) => {
  const [history, setHistory] = useState([]);
  
  useEffect(() => {
    loadHistory();
  }, []);
  
  const loadHistory = () => {
    const historyData = JSON.parse(localStorage.getItem('mental_health_predictor_history') || '[]');
    
    // Sort by timestamp (newest first)
    historyData.sort((a, b) => b.timestamp - a.timestamp);
    
    setHistory(historyData);
  };
  
  if (history.length === 0) {
    return (
      <div className="history-section">
        <h2>Previous Analyses</h2>
        <div className="history-list">
          <div className="history-item">No previous analyses found.</div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="history-section">
      <h2>Previous Analyses</h2>
      <div className="history-list">
        {history.map((item, index) => {
          const date = new Date(item.timestamp);
          const formattedDate = date.toLocaleString();
          
          return (
            <div 
              key={index} 
              className="history-item" 
              onClick={() => onSelectItem(item)}
            >
              <div className="timestamp">{formattedDate}</div>
              <div className="preview">
                {item.text.substring(0, 50)}
                {item.text.length > 50 ? '...' : ''}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default HistorySection;