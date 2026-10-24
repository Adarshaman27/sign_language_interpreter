This directory contains the TensorFlow.js model served by the frontend.
model.json and the *.bin weight files are generated from
training/isl_model_clean.keras — they are binary/derived artifacts,
not something to hand-edit.

Generate them with (see main README.md for full steps):

    cd training
    pip install -r requirements.txt --break-system-packages
    tensorflowjs_converter --input_format keras isl_model_clean.keras ../frontend/model

After that this folder will contain model.json and one or more
group1-shard*.bin files, and the backend/frontend will work as-is.
