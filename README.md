# AI Data Visualization and Analysis

## Overview
This is an AI-powered data visualization and analysis tool built using **Streamlit**. It allows users to upload datasets, ask questions about their data, and receive AI-generated insights with visualizations. The tool leverages **Together AI's LLM models** for natural language processing and **E2B's sandboxed Python execution** for running data analysis code securely.

## Features
- **Upload CSV files** for data analysis.
- **AI-powered query processing** to extract insights from data.
- **Automated Python code execution** for analysis and visualization.
- **Dynamic data visualization** using Matplotlib and Seaborn.
- **Supports multiple AI models**, including Meta-Llama and DeepSeek.

## Installation

### Prerequisites
- Python 3.8+
- Streamlit
- Pandas
- Together AI API key
- E2B API key

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-data-viz.git
   cd ai-data-viz
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. **Enter API keys** for Together AI and E2B in the sidebar.
2. **Upload a CSV dataset**.
3. **Ask a question** about the dataset.
4. **Get AI-generated insights and visualizations**.

## API Configuration
- **Together AI API Key**: Required for LLM-based responses.
- **E2B API Key**: Required for executing Python code securely.
- **Model Selection**: Choose from different AI models for analysis.

## Troubleshooting
- **API Key Errors**: Ensure correct API keys are entered.
- **Dataset Issues**: Verify that the uploaded file is a valid CSV.
- **Visualization Not Displaying**: Check column names and data format.

## Future Enhancements
- **Support for additional file formats** (Excel, JSON).
- **Advanced ML-based analysis** for deeper insights.
- **Custom visualization selection** by users.

## License
This project is open-source under the MIT License.

---

### Contributors
- **[Your Name]** - Developer

Feel free to modify this README to fit your specific use case! 🚀


Kannan reference :https://github.com/GURPREETKAURJETHRA/AI-Data-Visualization-Agent

