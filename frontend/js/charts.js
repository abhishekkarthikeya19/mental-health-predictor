/**
 * Charts and visualizations for the Mental Health Predictor Dashboard
 */

// Chart objects
let sentimentChart = null;
let emotionChart = null;

// Initialize charts when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
});

/**
 * Initialize all charts with default data
 */
function initializeCharts() {
    createSentimentChart();
    createEmotionChart();
}

/**
 * Create the sentiment trend chart
 */
function createSentimentChart() {
    const ctx = document.getElementById('sentiment-chart').getContext('2d');
    
    sentimentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['7 days ago', '6 days ago', '5 days ago', '4 days ago', '3 days ago', '2 days ago', 'Today'],
            datasets: [{
                label: 'Sentiment Score',
                data: [0.2, 0.3, -0.1, -0.4, -0.5, -0.2, 0.1],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Sentiment Trend Over Time',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                const value = context.parsed.y;
                                let sentiment = 'Neutral';
                                if (value > 0.3) sentiment = 'Positive';
                                if (value < -0.3) sentiment = 'Negative';
                                label += `${value.toFixed(2)} (${sentiment})`;
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: -1,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Sentiment Score (-1 to 1)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });
}

/**
 * Create the emotion analysis chart
 */
function createEmotionChart() {
    const ctx = document.getElementById('emotion-chart').getContext('2d');
    
    emotionChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Anxiety', 'Sadness', 'Anger', 'Fear', 'Joy', 'Surprise'],
            datasets: [{
                label: 'Current Analysis',
                data: [0.4, 0.6, 0.3, 0.5, 0.2, 0.1],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }, {
                label: 'Previous Analysis',
                data: [0.3, 0.4, 0.2, 0.3, 0.5, 0.3],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgb(255, 99, 132)',
                pointBackgroundColor: 'rgb(255, 99, 132)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(255, 99, 132)'
            }]
        },
        options: {
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Emotional Analysis',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

/**
 * Update charts based on analysis history
 */
function updateCharts() {
    updateSentimentChart();
    updateEmotionChart();
}

/**
 * Update the sentiment trend chart with current data
 */
function updateSentimentChart() {
    if (!sentimentChart || analysisHistory.length === 0) return;
    
    // Prepare data for chart
    const labels = [];
    const data = [];
    
    // Sort history by timestamp
    const sortedHistory = [...analysisHistory].sort((a, b) => a.timestamp - b.timestamp);
    
    // Get data points
    sortedHistory.forEach(entry => {
        // Format date for label
        const date = new Date(entry.timestamp);
        const label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        
        // Calculate sentiment score (-1 to 1 scale)
        // Convert prediction (0 or 1) and confidence to a sentiment score
        // 0 (no distress) with high confidence = positive sentiment
        // 1 (distress) with high confidence = negative sentiment
        let sentimentScore;
        if (entry.prediction === 0) {
            sentimentScore = entry.confidence; // Positive score for no distress
        } else {
            sentimentScore = -entry.confidence; // Negative score for distress
        }
        
        labels.push(label);
        data.push(sentimentScore);
    });
    
    // Update chart data
    sentimentChart.data.labels = labels;
    sentimentChart.data.datasets[0].data = data;
    sentimentChart.update();
}

/**
 * Update the emotion analysis chart with current data
 */
function updateEmotionChart() {
    if (!emotionChart || analysisHistory.length === 0) return;
    
    // Get the most recent and second most recent entries
    const current = analysisHistory[0];
    const previous = analysisHistory.length > 1 ? analysisHistory[1] : null;
    
    // Generate simulated emotion data based on prediction and confidence
    // In a real application, this would come from a more sophisticated emotion analysis
    const currentEmotions = generateEmotionData(current);
    
    // Update current analysis dataset
    emotionChart.data.datasets[0].data = [
        currentEmotions.anxiety,
        currentEmotions.sadness,
        currentEmotions.anger,
        currentEmotions.fear,
        currentEmotions.joy,
        currentEmotions.surprise
    ];
    
    // Update previous analysis dataset if available
    if (previous) {
        const previousEmotions = generateEmotionData(previous);
        emotionChart.data.datasets[1].data = [
            previousEmotions.anxiety,
            previousEmotions.sadness,
            previousEmotions.anger,
            previousEmotions.fear,
            previousEmotions.joy,
            previousEmotions.surprise
        ];
        emotionChart.data.datasets[1].label = 'Previous Analysis';
    } else {
        // If no previous analysis, use baseline values
        emotionChart.data.datasets[1].data = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        emotionChart.data.datasets[1].label = 'Baseline';
    }
    
    emotionChart.update();
}

/**
 * Generate simulated emotion data based on prediction and confidence
 * @param {Object} entry - Analysis history entry
 * @returns {Object} - Emotion data object
 */
function generateEmotionData(entry) {
    const emotions = {
        anxiety: 0,
        sadness: 0,
        anger: 0,
        fear: 0,
        joy: 0,
        surprise: 0
    };
    
    // Randomize with some constraints based on prediction
    const randomFactor = 0.2; // Add some randomness
    
    if (entry.prediction === 1) {
        // Distress detected - higher negative emotions
        emotions.anxiety = 0.5 + (entry.confidence * 0.3) + (Math.random() * randomFactor);
        emotions.sadness = 0.4 + (entry.confidence * 0.4) + (Math.random() * randomFactor);
        emotions.anger = 0.2 + (entry.confidence * 0.2) + (Math.random() * randomFactor);
        emotions.fear = 0.3 + (entry.confidence * 0.3) + (Math.random() * randomFactor);
        emotions.joy = 0.3 - (entry.confidence * 0.2) + (Math.random() * randomFactor);
        emotions.surprise = 0.2 + (Math.random() * randomFactor);
    } else {
        // No distress - higher positive emotions
        emotions.anxiety = 0.3 - (entry.confidence * 0.1) + (Math.random() * randomFactor);
        emotions.sadness = 0.2 - (entry.confidence * 0.1) + (Math.random() * randomFactor);
        emotions.anger = 0.1 + (Math.random() * randomFactor);
        emotions.fear = 0.2 + (Math.random() * randomFactor);
        emotions.joy = 0.5 + (entry.confidence * 0.3) + (Math.random() * randomFactor);
        emotions.surprise = 0.3 + (Math.random() * randomFactor);
    }
    
    // Ensure values are between 0 and 1
    Object.keys(emotions).forEach(key => {
        emotions[key] = Math.max(0, Math.min(1, emotions[key]));
    });
    
    return emotions;
}