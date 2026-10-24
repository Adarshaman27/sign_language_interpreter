# SignBridge — ISL to Speech Interpreter

Converts Indian Sign Language hand gestures (webcam + MediaPipe) into words,
then uses Gemini to turn the recognized words into a natural English sentence,
and reads it aloud.



## Project structure

```
signbridge/
├── frontend/           # served by the Flask backend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── .env.example
├── model/               # generated - see Step 1 below
│   └── README.txt
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

