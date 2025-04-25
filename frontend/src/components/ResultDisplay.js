import React from 'react';

const ResultDisplay = ({ result }) => {
  if (!result) return null;
  
  const className = `result ${result.isDistressed ? 'distressed' : 'normal'}`;
  
  return (
    <div className={className} role="alert">
      {result.text}
    </div>
  );
};

export default ResultDisplay;