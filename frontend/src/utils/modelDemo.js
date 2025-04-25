/**
 * Model Demo Utility
 * 
 * This file provides utility functions to demonstrate the mental health prediction model
 * in the frontend, including example data and mock predictions for development purposes.
 */

// Example dataset entries for demonstration
export const exampleDataset = [
  {
    text: "I feel sad and empty inside all the time",
    label: 1, // Distressed
    source: "Custom Dataset"
  },
  {
    text: "I'm so depressed I can barely get out of bed most days",
    label: 1, // Distressed
    source: "Custom Dataset"
  },
  {
    text: "Nothing brings me joy anymore, everything feels pointless",
    label: 1, // Distressed
    source: "Custom Dataset"
  },
  {
    text: "Life is good, I'm enjoying my day",
    label: 0, // Normal
    source: "Custom Dataset"
  },
  {
    text: "I had a productive meeting at work today",
    label: 0, // Normal
    source: "Custom Dataset"
  },
  {
    text: "Feeling great after my workout this morning",
    label: 0, // Normal
    source: "Custom Dataset"
  },
  {
    text: "I feel like a burden to everyone around me",
    label: 1, // Distressed
    source: "Emotion Dataset"
  },
  {
    text: "I'm constantly anxious and can't relax for even a moment",
    label: 1, // Distressed
    source: "Emotion Dataset"
  },
  {
    text: "I'm excited about my upcoming vacation next month",
    label: 0, // Normal
    source: "Tweet Dataset"
  },
  {
    text: "Just finished a good book and feeling satisfied",
    label: 0, // Normal
    source: "Tweet Dataset"
  }
];

// Mock function to demonstrate model prediction
export const mockPredict = (text) => {
  // Simple keyword-based mock prediction for demonstration
  const distressKeywords = [
    'sad', 'depressed', 'anxious', 'hopeless', 'worthless', 
    'suicidal', 'empty', 'tired', 'exhausted', 'burden',
    'crying', 'alone', 'lonely', 'fear', 'scared', 'panic',
    'stress', 'overwhelm', 'pain', 'hurt', 'numb'
  ];
  
  const text_lower = text.toLowerCase();
  
  // Count distress keywords
  let keywordCount = 0;
  distressKeywords.forEach(keyword => {
    if (text_lower.includes(keyword)) {
      keywordCount++;
    }
  });
  
  // Calculate mock probability based on keyword count
  const distressProbability = Math.min(0.5 + (keywordCount * 0.1), 0.95);
  
  // Determine prediction (1 for distressed, 0 for normal)
  const prediction = distressProbability > 0.5 ? 1 : 0;
  
  // Create mock recommendation
  let recommendation = "";
  if (prediction === 1) {
    recommendation = "Based on your text, you may be experiencing some emotional distress. Consider talking to someone you trust or a mental health professional.";
  } else {
    recommendation = "Your text doesn't show significant signs of emotional distress. Continue practicing self-care and maintaining your mental wellbeing.";
  }
  
  // Return mock prediction result
  return {
    prediction,
    confidence: prediction === 1 ? distressProbability : 1 - distressProbability,
    recommendation
  };
};

// Function to get dataset statistics for visualization
export const getDatasetStats = () => {
  return {
    totalSamples: 5000, // Mock total samples
    trainingSamples: 4000,
    testingSamples: 1000,
    classCounts: {
      normal: 2500,
      distressed: 2500
    },
    sources: [
      { name: "Emotion Dataset", count: 2000 },
      { name: "Tweet Dataset", count: 2000 },
      { name: "Custom Dataset", count: 1000 }
    ],
    topWords: [
      { text: "feeling", count: 120 },
      { text: "today", count: 100 },
      { text: "better", count: 80 },
      { text: "anxiety", count: 70 },
      { text: "tired", count: 60 },
      { text: "happy", count: 50 },
      { text: "stressed", count: 50 },
      { text: "sleep", count: 40 },
      { text: "friends", count: 40 },
      { text: "work", count: 30 }
    ]
  };
};

// Function to get model performance metrics
export const getModelPerformance = () => {
  return {
    accuracy: 0.89,
    precision: 0.87,
    recall: 0.91,
    f1Score: 0.89,
    confusionMatrix: {
      truePositives: 450,
      falsePositives: 67,
      trueNegatives: 433,
      falseNegatives: 50
    },
    trainingTime: "45 minutes",
    modelSize: "125 MB"
  };
};