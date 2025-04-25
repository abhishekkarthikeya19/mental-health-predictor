import React, { useState, useEffect, useRef } from 'react';
import HistorySection from './components/HistorySection';
import ResultDisplay from './components/ResultDisplay';
import Dashboard from './components/Dashboard';

const App = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const textInputRef = useRef(null);
  
  // API Configuration
  const API_CONFIG = {
    baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? 'http://localhost:8000' 
      : 'https://mental-health-predictor-api.onrender.com',
    endpoints: {
      predict: '/predict/',
      analyze: '/analyze/'
    }
  };
  
  useEffect(() => {
    // Check for saved draft
    const savedDraft = localStorage.getItem('mental_health_predictor_draft');
    if (savedDraft) {
      setText(savedDraft);
    }
    
    // Focus the text input on load
    if (textInputRef.current) {
      textInputRef.current.focus();
    }
    
    // Load analysis data if available
    const savedAnalysisData = localStorage.getItem('mental_health_analysis_data');
    if (savedAnalysisData) {
      try {
        setAnalysisData(JSON.parse(savedAnalysisData));
      } catch (e) {
        console.error('Error parsing saved analysis data:', e);
      }
    }
  }, []);
  
  // Auto-save draft as user types
  useEffect(() => {
    localStorage.setItem('mental_health_predictor_draft', text);
  }, [text]);
  
  const handleTextChange = (e) => {
    setText(e.target.value);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };
  
  const clearForm = () => {
    setText('');
    setResult(null);
    setError('');
    localStorage.removeItem('mental_health_predictor_draft');
    if (textInputRef.current) {
      textInputRef.current.focus();
    }
  };
  
  const toggleHistory = () => {
    setShowHistory(!showHistory);
    if (!showHistory) {
      setShowDashboard(false);
    }
  };
  
  const toggleDashboard = () => {
    setShowDashboard(!showDashboard);
    if (!showDashboard) {
      setShowHistory(false);
    }
  };
  
  const saveToHistory = (text, prediction, resultText, confidence) => {
    const history = JSON.parse(localStorage.getItem('mental_health_predictor_history') || '[]');
    
    // Add new item
    const newItem = {
      text,
      prediction,
      result: resultText,
      confidence: confidence || 0,
      timestamp: Date.now()
    };
    
    history.push(newItem);
    
    // Keep only the most recent 10 items
    if (history.length > 10) {
      history.sort((a, b) => b.timestamp - a.timestamp);
      history.splice(10);
    }
    
    localStorage.setItem('mental_health_predictor_history', JSON.stringify(history));
    
    // Update analysis data with new prediction history
    updateAnalysisData(history);
  };
  
  const updateAnalysisData = (history) => {
    // Create or update analysis data based on history
    const dates = [];
    const confidence = [];
    const predictions = [];
    
    // Sort history by timestamp (oldest first)
    const sortedHistory = [...history].sort((a, b) => a.timestamp - b.timestamp);
    
    sortedHistory.forEach(item => {
      const date = new Date(item.timestamp);
      dates.push(date.toLocaleDateString());
      confidence.push(item.confidence);
      predictions.push(item.prediction);
    });
    
    // Calculate sentiment trends (simplified mock data)
    const sentimentTrends = {
      dates: dates,
      positive: sortedHistory.map(item => item.prediction ? 0.3 : 0.7),
      negative: sortedHistory.map(item => item.prediction ? 0.7 : 0.3)
    };
    
    // Mock data for other visualizations
    const topWords = [
      { text: "feeling", count: 12 },
      { text: "today", count: 10 },
      { text: "better", count: 8 },
      { text: "anxiety", count: 7 },
      { text: "tired", count: 6 },
      { text: "happy", count: 5 },
      { text: "stressed", count: 5 },
      { text: "sleep", count: 4 },
      { text: "friends", count: 4 },
      { text: "work", count: 3 }
    ];
    
    const emotionDistribution = [
      { emotion: "Neutral", percentage: 35 },
      { emotion: "Anxiety", percentage: 25 },
      { emotion: "Sadness", percentage: 15 },
      { emotion: "Joy", percentage: 15 },
      { emotion: "Anger", percentage: 5 },
      { emotion: "Fear", percentage: 5 }
    ];
    
    const languagePatterns = {
      categories: ["First-person", "Negative words", "Question marks", "Exclamations", "Long sentences"],
      normal: [65, 20, 5, 10, 30],
      distressed: [80, 45, 15, 25, 50]
    };
    
    const userActivity = {
      dates: dates,
      counts: sortedHistory.map(() => Math.floor(Math.random() * 5) + 1)
    };
    
    const predictionHistory = {
      dates: dates,
      confidence: confidence
    };
    
    // Generate insights based on the data
    const insights = [];
    
    // Check for consistent negative sentiment
    const recentPredictions = predictions.slice(-3);
    if (recentPredictions.filter(p => p === 1).length >= 2) {
      insights.push({
        type: 'warning',
        text: 'Multiple recent entries show signs of emotional distress. Consider reaching out for support.'
      });
    }
    
    // Check for improvement
    if (predictions.length >= 5) {
      const olderPredictions = predictions.slice(0, 3);
      const newerPredictions = predictions.slice(-3);
      
      const olderDistressCount = olderPredictions.filter(p => p === 1).length;
      const newerDistressCount = newerPredictions.filter(p => p === 1).length;
      
      if (olderDistressCount > newerDistressCount) {
        insights.push({
          type: 'positive',
          text: 'Your recent entries show improvement in emotional well-being compared to earlier entries.'
        });
      }
    }
    
    // Add general insight
    insights.push({
      type: 'info',
      text: 'Regular journaling can help track emotional patterns and identify triggers for distress.'
    });
    
    const newAnalysisData = {
      sentimentTrends,
      topWords,
      emotionDistribution,
      languagePatterns,
      userActivity,
      predictionHistory,
      insights
    };
    
    setAnalysisData(newAnalysisData);
    localStorage.setItem('mental_health_analysis_data', JSON.stringify(newAnalysisData));
  };
  
  const handlePredict = async () => {
    const trimmedText = text.trim();
    
    // Validate input
    if (!trimmedText) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    if (trimmedText.length < 10) {
      setError('Please enter at least 10 characters for a meaningful analysis.');
      return;
    }
    
    setError('');
    setLoading(true);
    setResult(null);
    
    try {
      const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.predict}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_input: trimmedText })
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      
      // Create result text
      let resultText = data.prediction ? 
        'Analysis Result: Potential signs of distress detected' : 
        'Analysis Result: No significant signs of distress detected';
        
      // Add confidence score if available
      let confidencePercent = 0;
      if (data.confidence !== undefined) {
        confidencePercent = Math.round(data.confidence * 100);
        resultText += ` (Confidence: ${confidencePercent}%)`;
      }
      
      // Add recommendation if available
      if (data.recommendation) {
        resultText += `\n\n${data.recommendation}`;
      }
      
      const resultData = {
        text: resultText,
        isDistressed: data.prediction
      };
      
      setResult(resultData);
      
      // Save to history
      saveToHistory(trimmedText, data.prediction, resultText, data.confidence);
      
      // Clear draft since we've successfully analyzed it
      localStorage.removeItem('mental_health_predictor_draft');
      
      // Try to get detailed analysis if available
      try {
        const analysisResponse = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.analyze}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text_input: trimmedText })
        });
        
        if (analysisResponse.ok) {
          const analysisData = await analysisResponse.json();
          // Update analysis data with server response
          // This would be implemented if the backend supports detailed analysis
        }
      } catch (analysisError) {
        console.log('Detailed analysis not available:', analysisError);
        // Continue without detailed analysis
      }
      
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'An error occurred while analyzing the text. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const loadFromHistory = (historyItem) => {
    setText(historyItem.text);
    setResult({
      text: historyItem.result,
      isDistressed: historyItem.prediction
    });
    setShowHistory(false);
    window.scrollTo(0, 0);
  };
  
  return (
    <>
      <div className="container">
        <h1>Mental Health Prediction</h1>
        
        <div className="nav-tabs">
          <button 
            className={!showDashboard ? 'active' : ''} 
            onClick={() => setShowDashboard(false)}
          >
            Text Analysis
          </button>
          <button 
            className={showDashboard ? 'active' : ''} 
            onClick={() => setShowDashboard(true)}
          >
            Dashboard
          </button>
        </div>
        
        {!showDashboard ? (
          <>
            <div className="input-group">
              <label htmlFor="text_input">How are you feeling?</label>
              <textarea 
                id="text_input" 
                ref={textInputRef}
                value={text}
                onChange={handleTextChange}
                onKeyPress={handleKeyPress}
                placeholder="Describe how you're feeling or what's on your mind..." 
                aria-describedby="text_input_help"
                maxLength={5000}
                aria-required="true"
              />
              <span id="text_input_help" className="character-counter">
                {text.length}/5000 characters
              </span>
            </div>
            
            <div className="button-group">
              <button 
                id="predict_button" 
                onClick={handlePredict} 
                disabled={loading}
                aria-label="Analyze text for mental health indicators"
              >
                Analyze Text
              </button>
              <button 
                id="clear_button" 
                onClick={clearForm} 
                disabled={loading}
                className="secondary"
                aria-label="Clear form and results"
              >
                Clear
              </button>
            </div>
            
            {error && (
              <div className="error" role="alert">
                {error}
              </div>
            )}
            
            {loading && (
              <div className="loading" aria-live="polite">
                <div className="spinner" aria-hidden="true"></div>
                <p>Analyzing your text...</p>
              </div>
            )}
            
            {result && <ResultDisplay result={result} />}
            
            {showHistory && (
              <HistorySection onSelectItem={loadFromHistory} />
            )}
          </>
        ) : (
          <Dashboard analysisData={analysisData} />
        )}
        
        <div className="disclaimer">
          <strong>Disclaimer:</strong> This tool is for educational purposes only and is not a substitute for professional mental health advice. If you or someone you know is experiencing a mental health crisis, please contact a mental health professional or a crisis helpline immediately.
        </div>
      </div>
      
      <div className="footer">
        <p>
          Mental Health Predictor &copy; {new Date().getFullYear()} | 
          <button 
            className="link-button" 
            onClick={toggleHistory}
            disabled={showDashboard}
          >
            {showHistory ? 'Hide History' : 'View History'}
          </button>
        </p>
      </div>
    </>
  );
};

export default App;
