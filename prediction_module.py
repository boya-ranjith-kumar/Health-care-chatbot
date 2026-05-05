import pickle
import numpy as np
from nlp_processing import preprocess_symptoms, get_disease_info

# Load model & encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


def predict_disease(vector):
    vector = vector.reshape(1, -1)

    probs = model.predict_proba(vector)[0]
    idx = np.argmax(probs)

    disease = label_encoder.inverse_transform([idx])[0]
    confidence = probs[idx] * 100
    

    return disease, confidence


def get_full_response(user_input):
    vector, detected = preprocess_symptoms(user_input)
    
    if len(detected) == 0:
        return {
            "error": True,
            "message": "Sorry, I couldn’t detect any symptoms. Please enter your symptoms clearly (for example, fever, cough, headache)."
        }

    disease, confidence = predict_disease(vector)
    desc, prec = get_disease_info(disease)

    return {
        "disease": disease,
        "confidence": confidence,
        "description": desc,
        "precautions": prec,
        "detected_symptoms": detected,
        "count": len(detected)
    }