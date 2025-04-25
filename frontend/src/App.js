import React, { useState, useEffect, useRef } from 'react';
import HistorySection from './components/HistorySection';
import ResultDisplay from './components/ResultDisplay';

const App = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const textInputRef = useRef(null);
  
  // API Configuration
  const API_CONFIG = {
    baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? 'http://localhost:8000' 
      : 'https://your-production-api.com',
    endpoints: {
      predict: '/predict/'
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
  };
  
  const saveToHistory = (text, prediction, resultText) => {
    const history = JSON.parse(localStorage.getItem('mental_health_predictor_history') || '[]');
    
    // Add new item
    history.push({
      text,
      prediction,
      result: resultText,
      timestamp: Date.now()
    });
    
    // Keep only the most recent 10 items
    if (history.length > 10) {
      history.sort((a, b) => b.timestamp - a.timestamp);
      history.splice(10);
    }
    
    localStorage.setItem('mental_health_predictor_history', JSON.stringify(history));
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
      if (data.confidence !== undefined) {
        const confidencePercent = Math.round(data.confidence * 100);
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
      saveToHistory(trimmedText, data.prediction, resultText);
      
      // Clear draft since we've successfully analyzed it
      localStorage.removeItem('mental_health_predictor_draft');
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
          >
            {showHistory ? 'Hide History' : 'View History'}
          </button>
        </p>
      </div>
    </>
  );
};

export default App;