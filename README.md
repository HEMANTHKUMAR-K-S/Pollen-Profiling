# Pollen's Profiling 🌼

AI-powered web app that classifies pollen grains using deep learning.

## Features
- Upload image via web
- CNN-based prediction
- Clean Tailwind UI
- Deploy-ready (Render/Vercel)

## How to Run

```bash
# Train model (only once)
python train_model.py

# Start backend
cd flask
pip install -r requirements.txt
python app.py
```

Visit: http://localhost:5000

## Deployment

- Push to GitHub
- Use [https://render.com](https://render.com)
- Connect repo, set Build Command and Start Command
