# DermaAssist-DX
AI-powered dermatological analysis for Indian skin conditions (Fitzpatrick IV-VI)

## What it does
- Classifies skin conditions into 8 categories using Swin Transformer trained on DermaCon-IN
- Generates Grad-CAM heatmaps showing which skin regions influenced the prediction
- Produces structured clinical reasoning reports via LLaMA 4 Scout (Groq API)
- Confidence-gated safety mechanism — no clinical output below 50% confidence

## Tech Stack
PyTorch · Swin Transformer · Grad-CAM · Groq API · Streamlit · DermaCon-IN

## How to run
pip install -r requirements.txt
streamlit run app.py

## Results
68.32% validation accuracy · 5,450 Indian clinical images · 8 disease categories
