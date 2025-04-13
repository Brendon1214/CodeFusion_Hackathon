# CodeFusion Hackathon

## Overview

This project is a **flood disaster management system** designed for **CodeFusion Hackathon**. The app allows users to submit flood disaster reports, view statistics, generate heatmaps, ask questions about flood data, predict flood risks using weather data, and receive notifications via Telegram. 

The system also supports real-time flood data analysis, with the help of language models and embeddings to provide answers based on historical reports.

### Features:

- **Submit Report**: Users can submit flood reports, including location, severity, image, and description.
- **Statistics Dashboard**: Visualizes flood report statistics (e.g., severity, location distribution).
- **Community Gallery**: Displays uploaded flood photos with optional descriptions.
- **Heatmap**: Generates a heatmap visualization of flood reports based on location and severity.
- **RAG Chat**: Ask about flood reports (e.g., which locations had severe floods).
- **Flood Risk Prediction**: Predicts flood risk based on weather data and water levels, with Telegram notifications.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Brendon1214/CodeFusion_Hackathon.git
   cd CodeFusion_Hackathon

Install the required dependencies:

bash
复制
编辑
pip install -r requirements.txt
The requirements.txt file includes the following packages:

streamlit: For building the web interface.

pandas: For handling data operations.

plotly: For generating interactive visualizations.

pydeck: For displaying maps.

joblib: For loading pre-trained machine learning models.

requests: For making API requests (e.g., weather data).

langchain: For integrating language models and vector search.

transformers: For working with pre-trained language models.

chromadb: For vector database storage.

Ensure you have the appropriate model files (e.g., flood prediction model) and place them in the model/ directory.

Usage
Run the application:

bash
复制
编辑
streamlit run system.py
The application will start and be accessible at http://localhost:8501 in your browser.

Features Breakdown
Submit Report
Users can submit their flood reports, including the location, severity, an optional photo, and description.

Submitted reports are saved to a CSV file (disaster_reports.csv) for further analysis.

Statistics Dashboard
The dashboard displays flood report statistics such as the number of reports by severity and location.

It also shows a timeline of reports over time.

Community Gallery
The gallery displays images uploaded by users, showcasing the community's flood report submissions.

Users can browse through the gallery, view images, and delete photos if needed.

Heatmap
The heatmap visualizes the geographic distribution of flood reports, with intensity based on the severity of the floods.

RAG Chat
The RAG (Retrieval-Augmented Generation) chat allows users to ask questions about the flood data (e.g., "Which locations had severe floods?").

It leverages LangChain and language models for processing the questions and providing answers.

Flood Risk Prediction
Based on weather data (e.g., rainfall forecasts), the system predicts the likelihood of flooding in specific locations.

The system uses a pre-trained flood prediction model to make predictions based on simulated water levels and alerts the user via Telegram.

Example of Telegram Alert
When a high flood risk is predicted, a Telegram notification will be sent to the designated chat group, alerting the users to take precautionary measures.

Technologies Used
Streamlit: Web framework for building interactive data applications.

LangChain: Library for integrating language models with data retrieval.

Chroma: Vector database for storing and retrieving document embeddings.

Transformers: Library by Hugging Face for using pre-trained language models.

PyDeck: For map visualizations (heatmaps).

Plotly: For interactive data visualizations.

Contributing
Feel free to fork the repository and submit pull requests with improvements or new features. If you encounter any issues or bugs, please open an issue, and I will address them as soon as possible.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Special thanks to LangChain and Hugging Face for providing the libraries and models.

Thanks to the developers of Streamlit, Plotly, Chroma, and PyDeck for their contributions to making the application more powerful.

