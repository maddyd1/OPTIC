#!/bin/bash

# 1. Automatically find and go to the folder where this script lives
cd "$(dirname "$0")"

# 2. Turn on the project bubble (virtual environment)
source venv/bin/activate

# 3. Launch the dashboard
streamlit run app.py
