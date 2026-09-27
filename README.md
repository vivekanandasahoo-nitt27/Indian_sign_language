# Indian Sign Language Live Translator

This is the live application for the trained 229-class Indian Sign Language model.

## Model files

Place these files inside:

D:\PROJECTS\Indian_sign_language\indian_sign_model\

- isl_229_best_external_test.keras
- scaler.pkl
- classes.json

## Pipeline

Browser camera
-> sample approximately 20 frames / 3 seconds
-> Flask
-> MediaPipe Hands + Pose
-> 135 features
-> StandardScaler
-> 229-class Keras model
-> confidence >= 70%
-> same prediction for 5 sampled frames
-> confirmed sign
-> rolling 20-second sentence

## Run

PowerShell:

cd D:\PROJECTS\Indian_sign_language

conda activate hand_sign39

pip install -r requirements.txt

python app.py

Open:

http://127.0.0.1:5000

## Feature order

The backend uses the same 135-feature layout as training:

- left hand = 63
- right hand = 63
- left shoulder = 3
- right shoulder = 3
- left_present = 1
- right_present = 1
- shoulder_present = 1

Total = 135.

## Temporal logic

A sign is confirmed after five consecutive sampled predictions with confidence >= 70%.

After confirmation, the same held sign is not repeatedly added. A different/low-confidence sequence releases the previous sign.

The sentence uses a rolling 20-second window. Words older than 20 seconds disappear automatically.
