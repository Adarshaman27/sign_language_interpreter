import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PATHS
# ============================================================

# Load backend/.env specifically
load_dotenv(Path(__file__).resolve().parent / ".env")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNING: GEMINI_API_KEY not found.")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "gemini_configured": GEMINI_API_KEY is not None
    })


# ============================================================
# GEMINI SENTENCE IMPROVEMENT
# ============================================================

@app.route("/api/improve", methods=["POST"])
def improve_sentence():

    if not request.is_json:
        return jsonify({
            "error": "Request must contain JSON"
        }), 400

    data = request.get_json()

    words = data.get("words", [])

    if not words:
        return jsonify({
            "sentence": ""
        })

    if not client:
        return jsonify({
            "sentence": " ".join(words)
        })

    raw_text = " ".join(words)

    prompt = f"""
You are helping an Indian Sign Language communication system.

The system recognized these sign-language words, in this order:

{raw_text}

Convert them into one natural, simple English sentence.

Rules:
- Preserve the original meaning.
- Do not invent information that wasn't in the words.
- Fix grammar, tense, and word order.
- Keep the sentence short and conversational.
- Do not explain your answer.
- Return only the final sentence, nothing else.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=100,
            ),
        )

        sentence = (response.text or raw_text).strip()

        return jsonify({
            "sentence": sentence
        })

    except Exception as e:

        print("Gemini error:", e)

        return jsonify({
            "sentence": raw_text,
            "error": str(e)
        })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SIGNBRIDGE SERVER")
    print("=" * 60)
    print("Website: http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/api/health")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
