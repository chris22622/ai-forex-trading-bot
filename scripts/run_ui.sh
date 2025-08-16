#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
streamlit run ui/streamlit_app.py
