from fastapi import APIRouter
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder

router = APIRouter()
encoder = LabelEncoder()

rf_model = pickle.load(open("Trained-Models/rf_model.pkl", "rb"))
nb_model = pickle.load(open("Trained-Models/nb_model.pkl", "rb"))  
svm_model = pickle.load(open("Trained-Models/svm_model.pkl", "rb"))

DATA_PATH = "data/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
symptoms = X.columns.values

symptom_index = {" ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms)}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Disease to Specialist Mapping
disease_to_specialist = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist/Immunologist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Hepatologist',
    'Drug Reaction': 'Allergist/Immunologist',
    'Peptic ulcer diseae': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist',
    'Diabetes ': 'Diabetologist',
    'Gastroenteritis': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
    'Hypertension ': 'Cardiologist',
    'Migraine': 'Neurologist',
    'Cervical spondylosis': 'Orthopedic Specialist',
    'Paralysis (brain hemorrhage)': 'Neurologist',
    'Jaundice': 'Hepatologist',
    'Malaria': 'Infectious Disease Specialist',
    'Chicken pox': 'Infectious Disease Specialist',
    'Dengue': 'Infectious Disease Specialist',
    'Typhoid': 'Infectious Disease Specialist',
    'hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Alcoholic hepatitis': 'Hepatologist',
    'Tuberculosis': 'Pulmonologist',
    'Common Cold': 'General Practitioner',
    'Pneumonia': 'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Colorectal Surgeon',
    'Heart attack': 'Cardiologist',
    'Varicose veins': 'Vascular Surgeon',
    'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypoglycemia': 'Endocrinologist',
    'Osteoarthristis': 'Rheumatologist',
    'Arthritis': 'Rheumatologist',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurologist',
    'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist',
    'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist'
}

# Hospitals Data
hospitals = {
    "Abu Dhabi": [
        {"name": "Sheikh Khalifa Medical City", "departments": ["Cardiology", "Neurology", "Urology", "Pediatrics", "Radiology", "Surgery"]},
        {"name": "Tawam Hospital", "departments": ["Oncology", "Urology", "Neurosurgery", "Emergency Medicine", "Internal Medicine"]},
        {"name": "Al Mafraq Hospital", "departments": ["General Surgery", "Orthopedics", "Pediatrics", "Radiology", "Emergency Medicine"]},
        {"name":"Al Rahba Hospital", "departments": ["Neurosurgery", "Urology", "Psychiatry", "Family Medicine", "Emergency Medicine", "Internal Medicine", "Pediatrics", "Obstetrics", "Gynecology"]},
        {"name": "Corniche Hospital",    "departments": ["Obstetrics", "Gynecology", "Neonatology", "Maternal-Fetal Medicine", "Lactation Consultancy", "Anesthesiology", "Emergency Medicine"]},
        {
            "name": "Cleveland Clinic Abu Dhabi",
            "departments": [
                "Allergy and Immunology", "Anesthesiology", "Bariatric Surgery", "Cardiology",
                "Cardiovascular Medicine", "Colonoscopy", "Cosmetic Surgery", "Critical Care Medicine",
                "Dermatology", "Diabetes Program", "Digestive Diseases", "Emergency Department",
                "Endocrinology", "Endoscopy", "Epilepsy", "Executive Health",
                "Eye Care", "Facial Plastic and Reconstructive Surgery", "Family Medicine",
                "Gastroenterology", "General Internal Medicine", "General Surgery",
                "Head and Neck Surgery", "Heart and Vascular Surgery", "Hematology",
                "Infectious Diseases", "Nephrology", "Neurology", "Neurosurgery",
                "Obstetrics and Gynecology", "Oncology", "Orthopedic Surgery",
                "Pain Management", "Pediatrics", "Physical Medicine and Rehabilitation",
                "Plastic Surgery", "Psychiatry", "Pulmonary Medicine", "Radiology",
                "Rheumatology", "Sleep Medicine", "Sports Medicine", "Thoracic Surgery",
                "Transplant Surgery", "Urology", "Vascular Surgery"
            ]
        },
        {"name": "Burjeel Hospital", "departments": ["Cardiology", "Orthopedics", "Neurosurgery", "Gastroenterology", "Urology", "Pulmonology"]}
    ],
    "Dubai": [
        {"name": "Dubai Hospital", "departments": ["Cardiology", "Oncology", "Neurology", "Nephrology", "Urology", "Pulmonology"]},
        {"name": "American Hospital Dubai", "departments": ["Cardiology", "Dermatology", "Endocrinology", "Gastroenterology", "Neurology", "Urology"]},
        {"name": "Mediclinic City Hospital", "departments": ["Cardiology", "Dermatology", "Gastroenterology", "Neurology", "Pulmonology", "Rheumatology"]},
        {"name": "Saudi German Hospital Dubai", "departments": ["Cardiology", "Dermatology", "Endocrinology", "Gastroenterology", "Neurology", "Urology"]},
        {"name": "King's College Hospital London – Dubai", "departments": ["Cardiology", "Endocrinology", "Gastroenterology", "Neurology", "Pulmonology", "Urology"]}
    ],
    "Sharjah": [
        {"name": "Al Qassimi Hospital", "departments": ["Cardiology", "Neurology", "Pediatrics", "Radiology", "Urology", "Gastroenterology"]},
        {"name": "Al Zahra Hospital Sharjah", "departments": ["Cardiology", "Neurology", "Orthopedics", "Dermatology", "Gastroenterology", "Urology"]},
        {"name": "University Hospital Sharjah", "departments": ["Cardiology", "Neurology", "Pediatrics", "Rheumatology", "Pulmonology", "Urology"]}
    ],
    "Ajman": [
        {"name": "Sheikh Khalifa General Hospital", "departments": ["Cardiology", "Neurology", "Dermatology", "Urology", "Radiology", "Pulmonology"]}
    ],
    "Fujairah": [
        {"name": "Fujairah Hospital", "departments": ["Dermatology", "Digestive System", "Nephrology", "Radiology", "Pulmonology", "Urology"]},
        {"name": "Al Sharq Hospital", "departments": ["Cardiology", "Gastroenterology", "Nephrology", "Orthopedics", "Urology", "Pulmonology"]}
    ]
}

# Specialist to Department Mapping
specialist_to_department = {
    "Anesthesiologist": "Anesthesiology",
    "Allergist/Immunologist": "Allergy and Immunology",
    "Cardiologist": "Cardiology",
    "Colorectal Surgeon": "General Surgery",
    "Dermatologist": "Dermatology",
    "Diabetologist": "Diabetes Program",
    "Endocrinologist": "Endocrinology",
    "Epileptologist": "Epilepsy",
    "ENT Specialist": "ENT (Ear, Nose, and Throat)",
    "Gastroenterologist": "Gastroenterology",
    "General Practitioner": "General Medicine",
    "Gynecologist": "Gynecology",
    "Hepatologist": "Hepatology",
    "Hematologist": "Hematology",
    "Infectious Disease Specialist": "Infectious Diseases",
    "Pulmonologist": "Pulmonology",
    "Neurologist": "Neurology",
    "Nephrologist": "Nephrology",
    "Orthopedic Specialist": "Orthopedics",
    "Oncologist": "Oncology",
    # "Ophthalmologist": "Ophthalmology",
    "Ophthalmologist": "Eye Care",
    "Obstetrician": "Obstetrics",
    "Psychiatrist": "Psychiatry",
    "Pulmonologist": "Pulmonary Medicine",
    "Rheumatologist": "Rheumatology",
    "Radiologist": "Radiology",
    "Urologist": "Urology",
    "Vascular Surgeon": "Vascular Surgery",
}
class SymptomsRequest(BaseModel):
    symptoms: list[str]

@router.post("/predict")
def predict_specialist(request: SymptomsRequest):
    if request.symptoms is None:
        return {"message": "Symptoms not provided"}
    
    if request.symptoms == [""]:
        return {"message": "No symptoms provided"}
    
    input_data = [0] * len(symptom_index)
    print(request.symptoms)
    for symptom in request.symptoms:
        if symptom not in data_dict["symptom_index"]:
            continue
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    specialist = disease_to_specialist.get(final_prediction, "General Practitioner")
    # department = specialist_to_department.get(specialist, specialist)

    # available_hospitals = []
    # for city, city_hospitals in hospitals.items():
    #     for hospital in city_hospitals:
    #         if department in hospital["departments"]:
    #             available_hospitals.append({"city": city, "hospital": hospital["name"]})
    print("predicted_disease: ",final_prediction)
    print("specialist: ",specialist)
    return {
        "predicted_disease": final_prediction,
        "specialist": specialist,
        # "available_hospitals": available_hospitals
    }
