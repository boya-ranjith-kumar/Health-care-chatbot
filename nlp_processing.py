import pickle
import numpy as np
import pandas as pd

# Load symptom names
with open('symptoms.pkl', 'rb') as f:
    symptom_names = pickle.load(f)

# Loading Descriptions and Precautions datasets
desc_df = pd.read_csv(r'Cleaned datasets/Description data.csv')
prec_df = pd.read_csv(r'Cleaned datasets/Precautions data.csv')

# Converting Descriptions and Precautions to dictionary
disease_descriptions = dict(zip(desc_df.iloc[:, 0], desc_df.iloc[:, 1]))
disease_precautions = {}

for _, row in prec_df.iterrows():
    disease = row.iloc[0]
    precautions = ' '.join(row.iloc[1:].dropna().astype(str))
    disease_precautions[disease] = precautions

# some user input symptoms which doesn't match to symptoms in our dataset, 
# so each symptom in our data will be like root and similar words is placed to that root
symptom_keywords = {
     'itch': 'itching', 'itchy': 'itching',
    'rash': 'skin_rash', 'red': 'skin_rash', 'spots': 'skin_rash',
    'nodal': 'nodal_skin_eruptions', 'bumps': 'nodal_skin_eruptions',
    'sneeze': 'continuous_sneezing', 'sneezing': 'continuous_sneezing',
    'shiver': 'shivering', 'chill': 'chills',
    'joint': 'joint_pain', 'stomach': 'stomach_pain', 'pain': 'stomach_pain',
    'acid': 'acidity', 'heartburn': 'acidity', 'burn': 'acidity',
    'ulcer': 'ulcers_on_tongue', 'tongue': 'ulcers_on_tongue',
    'muscle waste': 'muscle_wasting', 'weak muscle': 'muscle_wasting',
    'vomit': 'vomiting', 'nausea': 'vomiting',
    'burn pee': 'burning_micturition', 'pee burn': 'burning_micturition',
    'blood urine': 'spotting_ urination',
    'tired': 'fatigue', 'exhaust': 'fatigue',
    'weight gain': 'weight_gain', 'fat': 'weight_gain',
    'anxiety': 'anxiety', 'panic': 'anxiety',
    'cold hands': 'cold_hands_and_feets', 'cold feet': 'cold_hands_and_feets',
    'mood swing': 'mood_swings', 'irritable': 'mood_swings',
    'weight loss': 'weight_loss', 'thin': 'weight_loss',
    'restless': 'restlessness', 'fidget': 'restlessness',
    'lethargy': 'lethargy', 'lazy': 'lethargy',
    'throat patch': 'patches_in_throat', 'white throat': 'patches_in_throat',
    'sugar': 'irregular_sugar_level', 'diabetes': 'irregular_sugar_level',
    'cough': 'cough',
    
    # FEVER & INFECTION
    'fever': 'high_fever', 'hot': 'high_fever', 'mild fever': 'mild_fever',
    'sunken eye': 'sunken_eyes',
    'breathless': 'breathlessness', 'short breath': 'breathlessness',
    'sweat': 'sweating', 'sweaty': 'sweating',
    'dehydrate': 'dehydration',
    'indigestion': 'indigestion',
    'headache': 'headache', 'head': 'headache',
    
    # DIGESTIVE (GERD/DIARRHEA)
    'yellow skin': 'yellowish_skin',
    'dark urine': 'dark_urine',
    'appetite loss': 'loss_of_appetite',
    'eye pain': 'pain_behind_the_eyes',
    'back': 'back_pain',
    'constipation': 'constipation',
    'abdominal': 'abdominal_pain', 'belly': 'abdominal_pain',
    'diarrhea': 'diarrhoea', 'loose stool': 'diarrhoea',
    'yellow urine': 'yellow_urine',
    'yellow eye': 'yellowing_of_eyes',
    
    # RESPIRATORY & EYE
    'phlegm': 'phlegm', 'mucus': 'phlegm',
    'throat irritate': 'throat_irritation',
    'red eye': 'redness_of_eyes',
    'sinus': 'sinus_pressure',
    'runny nose': 'runny_nose',
    'congestion': 'congestion',
    'chest pain': 'chest_pain',
    
    # MUSCLES & JOINTS
    'weak limb': 'weakness_in_limbs',
    'fast heart': 'fast_heart_rate',
    'anal pain': 'pain_in_anal_region',
    'bloody stool': 'bloody_stool',
    'neck': 'neck_pain',
    'dizzy': 'dizziness',
    'cramp': 'cramps',
    
    # THYROID & HORMONES
    'obesity': 'obesity',
    'swollen leg': 'swollen_legs',
    'puffy face': 'puffy_face_and_eyes',
    'thyroid': 'enlarged_thyroid',
    'brittle nail': 'brittle_nails',
    'hungry': 'excessive_hunger',
    
    # ARTHRITIS & NEURO
    'knee': 'knee_pain', 'hip': 'hip_joint_pain',
    'muscle weak': 'muscle_weakness',
    'stiff neck': 'stiff_neck',
    'swollen joint': 'swelling_joints',
    'stiff': 'movement_stiffness',
    'spin': 'spinning_movements',
    'balance': 'loss_of_balance',
    'one side weak': 'weakness_of_one_body_side',
    
    # SKIN & ALLERGY (CONTINUED)
    'internal itch': 'internal_itching',
    'toxic look': 'toxic_look_(typhos)',
    'depress': 'depression',
    'irritable': 'irritability',
    'muscle pain': 'muscle_pain',
    'red spots': 'red_spots_over_body',
    'belly pain': 'belly_pain',
    
    # EXTRA (Less common but important)
    'blister': 'blister',
    'blackhead': 'blackheads',
    'scalp': 'scurring',
    'skin peel': 'skin_peeling'
}

def preprocess_symptoms(user_input):
    user_input = user_input.lower()

    detected = []
    for key, value in symptom_keywords.items():
        if key in user_input:
            detected.append(value)

    # Create dynamic vector
    vector = np.zeros(len(symptom_names))

    for symptom in detected:
        if symptom in symptom_names:
            idx = symptom_names.index(symptom)
            vector[idx] = 1

    return vector, detected


def get_disease_info(disease):
    desc = disease_descriptions.get(disease, "Sorry! No description available")
    prec = disease_precautions.get(disease, "Consult a doctor")
    return desc, prec