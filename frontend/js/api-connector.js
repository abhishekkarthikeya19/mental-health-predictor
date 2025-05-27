/**
 * API Connector for Mental Health Predictor Dashboard
 * 
 * This file provides functions to connect the dashboard to the backend API.
 * To use this with the dashboard.html, add a script tag to include this file.
 */

// API base URL - change this to match your backend deployment
const API_BASE_URL = 'http://localhost:8000';

/**
 * Check the health of the API
 * @returns {Promise} Promise that resolves to the health check response
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error(`API health check failed with status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API Health Check Error:', error);
        throw error;
    }
}

/**
 * Analyze text using the prediction API
 * @param {string} text - The text to analyze
 * @returns {Promise} Promise that resolves to the prediction response
 */
async function analyzeTextWithApi(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text_input: text })
        });
        
        if (!response.ok) {
            throw new Error(`Analysis request failed with status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Analysis Error:', error);
        throw error;
    }
}

/**
 * Replace the demo analyzeText function in dashboard.html with this real API implementation
 * 
 * To use this function, replace the analyzeText function in dashboard.html with:
 * 
 * function analyzeText(text) {
 *     // Show loading state
 *     const resultContainer = document.getElementById('result-container');
 *     resultContainer.style.display = 'block';
 *     resultContainer.classList.add('loading');
 *     
 *     // Call the API
 *     analyzeTextWithApi(text)
 *         .then(data => {
 *             // Process the result
 *             processAnalysisResult(data, text);
 *         })
 *         .catch(error => {
 *             console.error('Analysis Error:', error);
 *             showError('Failed to analyze text. Please try again later.');
 *         })
 *         .finally(() => {
 *             // Remove loading state
 *             resultContainer.classList.remove('loading');
 *         });
 * }
 */

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        checkApiHealth,
        analyzeTextWithApi
    };
}