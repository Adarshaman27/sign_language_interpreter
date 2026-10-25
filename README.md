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

## Bing indexing

The frontend publishes a sitemap at `/sitemap.xml` and allows crawler access in
`/robots.txt`. After deploying changes, submit the homepage to Bing through
IndexNow:

```powershell
$body = @{
    host = "signlanguageinterpreter.vercel.app"
    key = "16a969c593754161a133071084f2e4cb"
    keyLocation = "https://signlanguageinterpreter.vercel.app/16a969c593754161a133071084f2e4cb.txt"
    urlList = @("https://signlanguageinterpreter.vercel.app/")
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://api.indexnow.org/indexnow" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

The key file must be publicly reachable at `keyLocation` before sending the
request. IndexNow notifies participating search engines; it does not guarantee
when or whether a URL will be indexed.
