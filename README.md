# Recommendation and Dialog System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-FF4B4B)

This project is a comprehensive solution combining a recommendation system and a dialogue system (chatbot). The interface is implemented using the Streamlit library.

## Description

The application allows users to interact with the system via a chat interface to receive personalized recommendations and responses to queries.

### Key Features

*   **Dialogue System:** Natural language processing (NLP) for user communication.
*   **Recommendation System:** Algorithms for matching content/items based on preferences.
*   **Interactive UI:** User-friendly web interface powered by Streamlit.

## Tech Stack

*   **Language:** Python 3.12
*   **UI Framework:** Streamlit
*   **ML Libraries:** scikit-learn, Transformers

## Installation and Setup

Follow the instructions below to deploy the project locally.

### Prerequisites

Ensure you have **Python 3.8** or higher installed.

### 1. Clone the repository

```bash
git clone https://github.com/northbae/RecommendationAndDialogSystem.git
cd RecommendationAndDialogSystem
```
### 2. Create a virtual environment (Recommended)
Windows:
```Bash
python -m venv venv
.\venv\Scripts\activate
```
macOS / Linux:
```Bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
You can install dependencies using requirements.txt:
```Bash
pip install -r requirements.txt
```
Or, since the project includes setup.py, install the project in editable mode:
```Bash
pip install -e .
```
## Usage
To launch the web interface, execute the following command from the project root:
```Bash
streamlit run src/ui/streamlit_ui.py --server.port 8080
```
After launching, the application will be available in your browser at:

http://localhost:8080
