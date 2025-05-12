from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import json


# Inicializa Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Modelos Pydantic para validação (equivalentes às suas interfaces TypeScript)
class CustomerData(BaseModel):
    name: str
    gender: Literal["male", "female"]
    age: int
    weight: float
    goal: Literal["hypertrophy", "weightLoss"]
    trainingTime: Literal["0-2", "2-6", "6-12", "12-17", "17+"]
    weeklyFrequency: Literal["2x", "3x", "5x+"]
    muscleFocus: Literal["general", "chest", "arms", "shoulders", "back", "lower", "quadriceps", "glutes", "hamstrings"]

class Exercise(BaseModel):
    id: str
    name: str
    sets: int
    repetitions: int
    category: Literal["chest", "arms", "shoulders", "back", "lower", "quadriceps", "glutes", "hamstrings", "mobility", "correction", "strengthening"]

class SubWorkout(BaseModel):
    name: str
    exercises: list[Exercise]

class Workout(BaseModel):
    id: str
    name: str
    workoutType: Literal["male", "female", "general"]
    difficulty: Literal["beginner", "intermediate1", "advanced"]
    goal: Literal["hypertrophy", "weightLoss"]
    category: Literal["general", "chest", "arms", "shoulders", "back", "lower", "quadriceps", "glutes", "hamstrings", "mobility", "correction", "strengthening"]
    weeklyFrequency: Literal["2x", "3x", "5x+"]
    subWorkouts: list[SubWorkout]

# Endpoint para receber dados do usuário e retornar o treino recomendado
@app.post("/recommend-workout")
async def recommend_workout(customer_data: CustomerData):
    try:
        # 1. Carregar dados de treinos do Firebase
        workouts_ref = db.collection("workouts")
        workouts = [doc.to_dict() for doc in workouts_ref.stream()]
        
        # 2. Pré-processar dados para ML
        df_workouts = pd.DataFrame(workouts)
        df_workouts = preprocess_workouts(df_workouts)
        
        # 3. Pré-processar dados do cliente
        client_features = preprocess_client(customer_data)
        
        # 4. Treinar/Usar modelo de recomendação
        recommended_workout_id = get_recommendation(df_workouts, client_features)
        

        return {"workout_id": recommended_workout_id }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Funções auxiliares
def preprocess_workouts(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados dos treinos"""
    required_columns = ['workoutType', 'difficulty', 'goal', 'category', 'weeklyFrequency', 'id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias faltando: {missing_cols}")

    encoders = {
        'workoutType': LabelEncoder(),
        'difficulty': LabelEncoder(),
        'goal': LabelEncoder(),
        'category': LabelEncoder(),
        'weeklyFrequency': LabelEncoder()
    }

    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])

    return df

def preprocess_client(client_data: CustomerData) -> dict:
    """Pré-processa os dados do cliente"""
    time_to_difficulty = {
        "0-2": 0, "2-6": 0, "6-12": 1, 
        "12-17": 1, "17+": 2  # 0=beginner, 1=intermediate1, 2=advanced
    }
    
    weekly_freq_mapping = {"2x": 0, "3x": 1, "5x+": 2}
    
    return {
        "gender": 0 if client_data.gender == "male" else 1,
        "age": client_data.age,
        "weight": client_data.weight,
        "goal": 0 if client_data.goal == "hypertrophy" else 1,
        "trainingTime": time_to_difficulty[client_data.trainingTime],
        "weeklyFrequency": weekly_freq_mapping[client_data.weeklyFrequency],
        "muscleFocus": client_data.muscleFocus.lower()
    }

def get_recommendation(workouts_df: pd.DataFrame, client_features: dict) -> str:
    """Lógica de recomendação com prioridade para weeklyFrequency"""
    try:
        # 1. Filtrar por frequência semanal primeiro
        freq_mask = workouts_df['weeklyFrequency'] == client_features['weeklyFrequency']
        matching_workouts = workouts_df[freq_mask]
        
        # Fallback: pegar a frequência mais próxima se não encontrar
        if len(matching_workouts) == 0:
            freq_diff = np.abs(workouts_df['weeklyFrequency'] - client_features['weeklyFrequency'])
            closest_idx = freq_diff.idxmin()
            matching_workouts = workouts_df.loc[[closest_idx]]

        # 2. Codificar features apenas com os treinos filtrados
        muscle_encoder = LabelEncoder()
        categories = list(matching_workouts['category'].unique()) + [client_features['muscleFocus']]
        muscle_encoder.fit(categories)

        difficulty_encoder = LabelEncoder()
        difficulties = list(matching_workouts['difficulty'].unique()) + [client_features['trainingTime']]
        difficulty_encoder.fit(difficulties)

        # 3. Preparar dados codificados
        matching_workouts = matching_workouts.copy()
        matching_workouts['category_encoded'] = muscle_encoder.transform(matching_workouts['category'])
        matching_workouts['difficulty_encoded'] = difficulty_encoder.transform(matching_workouts['difficulty'])

        client_vector = np.array([
            client_features['gender'],
            difficulty_encoder.transform([client_features['trainingTime']])[0],
            client_features['goal'],
            muscle_encoder.transform([client_features['muscleFocus']])[0],
            client_features['weeklyFrequency']
        ]).reshape(1, -1)

        workout_features = matching_workouts[[
            'workoutType',
            'difficulty_encoded',
            'goal',
            'category_encoded',
            'weeklyFrequency'
        ]].values

        # 4. Encontrar treino mais similar
        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(workout_features)
        distances, indices = knn.kneighbors(client_vector)

        return matching_workouts.iloc[indices[0][0]]['id']

    except Exception as e:
        raise ValueError(f"Erro na recomendação: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)