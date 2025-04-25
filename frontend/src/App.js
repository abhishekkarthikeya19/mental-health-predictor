import React, { useState } from 'react';

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [inputValue, setInputValue] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputValue }),
      });

      const data = await response.json();
      setResult(data.prediction || 'No prediction returned');
    } catch (error) {
      console.error('Error calling API:', error);
      setResult('Error occurred. Try again later.');
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>Mental Health Predictor</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          rows={6}
          cols={60}
          placeholder="Type something..."
        />
        <br />
        <button type="submit" style={{ marginTop: '10px' }}>
          Predict
        </button>
      </form>
      {result && (
        <div style={{ marginTop: '20px' }}>
          <strong>Prediction:</strong> {result}
        </div>
      )}
    </div>
  );
}

export default App;

