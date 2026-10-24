# SignBridge — ISL to Speech Interpreter

Converts Indian Sign Language hand gestures (webcam + MediaPipe) into words,
then uses Gemini to turn the recognized words into a natural English sentence,
and reads it aloud.



## Project structure

```
signbridge/
├── frontend/           # static Vercel frontend and browser assets
│   ├── index.html
│   ├── style.css
│   └── app.js
│   ├── model/           # TensorFlow.js model files
│   └── training/        # labels.json served to the browser
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── .env.example
└── training/
    ├── train.py
    ├── convert_model.py
    ├── convert_include50.py
    ├── requirements.txt
    ├── labels.json
    ├── isl_model.h5           # your already-trained weights
    ├── isl_model_clean.keras  # already-converted full model
    └── isl_dataset.json       # training data
```

Vercel serves `frontend/` at `/` and routes `/api/*` to the Flask service in
`backend/`. Run `vercel dev` from the repository root to test both services
together. The browser model files live under `frontend/model/`; `training/train.py`
also updates the browser-facing copy of `labels.json` under `frontend/training/`.

