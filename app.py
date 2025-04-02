import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Regex pattern to extract Python code from AI response
pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    """Executes Python code in an E2B sandbox and returns the result."""
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                execution = e2b_code_interpreter.run_code(code)
        # Capture errors
        if execution.error:
            st.error(f"Code Execution Error: {execution.error}")
            return None
        return execution.results

def match_code_blocks(llm_response: str) -> str:
    """Extracts Python code from the AI's response."""
    match = pattern.search(llm_response)
    return match.group(1) if match else ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    """Sends query to AI model and executes the returned Python code."""
    system_prompt = f"""
    You are a Python data scientist. The dataset is located at '{dataset_path}'.
    Analyze the dataset and provide Python code for the query.
    Always use the dataset path variable '{dataset_path}' when reading the CSV file.
    Please use the correct column names: 'Category' and 'Cost for two people' when grouping and plotting.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )
        response_message = response.choices[0].message.content
        python_code = match_code_blocks(response_message)
        if python_code:
            code_results = code_interpret(e2b_code_interpreter, python_code)
            return code_results, response_message
        else:
            st.warning("No valid Python code found in AI's response.")
            return None, response_message

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    """Uploads dataset to E2B sandbox."""
    # Reset pointer to start so that the file can be read properly
    uploaded_file.seek(0)
    dataset_path = f"./{uploaded_file.name}"
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error

def main():
    """Main Streamlit application."""
    st.title("📊 AI-Powered Data Analysis & Visualization")
    st.write("Upload a dataset and ask questions to analyze it!")

    # Initialize session state variables
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    # Sidebar for API keys and model selection
    with st.sidebar:
        st.header("🔑 API Keys & Model Selection")
        st.session_state.together_api_key = st.text_input("Together AI API Key", type="password")
        st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password")
        st.sidebar.markdown("[Get Together AI API Key](https://api.together.ai/signin)")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        selected_model = st.selectbox("Select Model", options=list(model_options.keys()))
        st.session_state.model_name = model_options[selected_model]

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")
    if uploaded_file:
        # Check if the uploaded file has data
        file_bytes = uploaded_file.read()
        if not file_bytes:
            st.error("Uploaded file is empty. Please upload a valid CSV file.")
            return
        # Reset pointer once to allow pandas to read from the beginning
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        st.write("### Dataset Preview:")
        show_full = st.checkbox("Show full dataset")
        st.dataframe(df if show_full else df.head())

        # Query input
        query = st.text_area("🔍 Ask a question about your dataset:",
                             "Can you compare the average cost for two people between different categories?")
        if st.button("Analyze Data"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("⚠️ Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset file to the sandbox
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)
                    st.write("### 🤖 AI Response:")
                    st.write(llm_response)
                    if code_results:
                        st.write("### 📈 Generated Insights:")
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:
                                png_data = base64.b64decode(result.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization")
                            elif hasattr(result, 'figure'):
                                st.pyplot(result.figure)
                            elif hasattr(result, 'show'):
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)

if __name__ == "__main__":
    main()
