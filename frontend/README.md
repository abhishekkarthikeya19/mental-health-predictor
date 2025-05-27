# Mental Health Predictor Dashboard

This directory contains a user-friendly visualization dashboard for the Mental Health Predictor project. The dashboard presents sentiment trends, predictions, and relevant insights from social media data, providing actionable insights to healthcare professionals.

## Features

- **Text Analysis Tool**: Analyze text input for signs of mental distress
- **Sentiment Trends**: Visualize sentiment changes over time
- **Emotional Analysis**: Radar chart showing different emotional dimensions
- **Early Warning Signs**: Identification of potential warning signs from text analysis
- **Intervention Recommendations**: Actionable insights for healthcare professionals
- **Mental Health Resources**: Links to crisis support, self-help resources, and professional help

## Files

- `dashboard.html`: The main dashboard file (includes HTML, CSS, and JavaScript)
- `js/api-connector.js`: Helper file to connect the dashboard to the backend API

## How to Use

1. **Open the Dashboard**:
   - Simply open the `dashboard.html` file in a web browser
   - For production use, deploy this file to a web server

2. **Connect to the Backend API**:
   - By default, the dashboard uses demo data
   - To connect to the real API, modify the dashboard to use the `api-connector.js` file
   - Update the `API_BASE_URL` in `api-connector.js` to point to your backend

3. **Analyze Text**:
   - Enter text in the analysis tool
   - Click "Analyze Text" to process the input
   - View the results and recommendations

4. **Interpret Visualizations**:
   - Sentiment Trends: Shows emotional state over time
   - Emotional Analysis: Breaks down different emotional dimensions
   - Warning Signs: Highlights potential areas of concern
   - Recommendations: Suggests next steps for intervention

## Integration with Backend

To integrate with the backend API:

1. Add this script tag to `dashboard.html`:
   ```html
   <script src="js/api-connector.js"></script>
   ```

2. Replace the demo `analyzeText` function in `dashboard.html` with the real API implementation as described in the comments of `api-connector.js`.

## Customization

The dashboard can be customized by:

- Modifying the CSS styles in the `<style>` section
- Updating the charts and visualizations in the JavaScript section
- Adding or removing sections based on specific requirements

## Disclaimer

This dashboard is for educational purposes only and is not a substitute for professional mental health advice. If you or someone you know is experiencing a mental health crisis, please contact a mental health professional or a crisis helpline immediately.