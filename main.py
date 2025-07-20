from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Inicializa Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Modelos Pydantic para validação
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

# Variáveis globais para armazenar encoders e scaler
encoders = {
    'workoutType': LabelEncoder(),
    'difficulty': LabelEncoder(),
    'goal': LabelEncoder(),
    'category': LabelEncoder(),
    'weeklyFrequency': LabelEncoder()
}
scaler = StandardScaler()

def load_and_preprocess_workouts():
    """Carrega e pré-processa os treinos uma vez ao iniciar"""
    workouts_ref = db.collection("workouts")
    workouts = [doc.to_dict() for doc in workouts_ref.stream()]
    
    if not workouts:
        raise ValueError("Nenhum treino encontrado no banco de dados")
    
    df = pd.DataFrame(workouts)
    
    # Verificar colunas obrigatórias
    required_columns = ['workoutType', 'difficulty', 'goal', 'category', 'weeklyFrequency', 'id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias faltando: {missing_cols}")
    
    # Codificar variáveis categóricas
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])
    
    # Selecionar features para o modelo
    features = df[['workoutType', 'difficulty', 'goal', 'category', 'weeklyFrequency']]
    
    # Normalizar os dados
    global scaler
    scaled_features = scaler.fit_transform(features)
    
    return df, scaled_features

# Carrega os dados ao iniciar a aplicação
try:
    workouts_df, workouts_features = load_and_preprocess_workouts()
except Exception as e:
    print(f"Erro ao carregar treinos: {str(e)}")
    workouts_df, workouts_features = None, None

@app.post("/recommend-workout")
async def recommend_workout(customer_data: CustomerData):
    try:
        if workouts_df is None:
            raise HTTPException(status_code=503, detail="Serviço de recomendação não disponível")
        
        # Pré-processar dados do cliente
        client_features = {
            "workoutType": customer_data.gender,
            "difficulty": time_to_difficulty(customer_data.trainingTime),
            "goal": customer_data.goal,
            "category": customer_data.muscleFocus,
            "weeklyFrequency": customer_data.weeklyFrequency
        }

        # Aplicar restrições de negócio
        if (client_features["weeklyFrequency"] == "2x" and 
            client_features["difficulty"] != "beginner"):
            # Se for intermediário/avançado pedindo 2x/semana, ajustamos para beginner
            client_features["difficulty"] = "beginner"        
        
        # Codificar features do cliente
        encoded_features = []
        for col in ['workoutType', 'difficulty', 'goal', 'category', 'weeklyFrequency']:
            encoded_features.append(encoders[col].transform([client_features[col]])[0])
        
        # Normalizar
        client_vector = scaler.transform([encoded_features])
        
        # Encontrar treinos mais similares
        knn = NearestNeighbors(n_neighbors=3, metric='cosine')
        knn.fit(workouts_features)
        distances, indices = knn.kneighbors(client_vector)
        
        # Pegar o mais similar (menor distância)
        best_match_idx = indices[0][0]
        recommended_workout = workouts_df.iloc[best_match_idx]
        
        return {
            "workout_id": recommended_workout['id'],
            "workout_name": recommended_workout['name'],
            "confidence": 1 - distances[0][0]  # Converter distância para "confiança"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na recomendação: {str(e)}")

def time_to_difficulty(training_time: str) -> str:
    """Mapeia tempo de treino para nível de dificuldade"""
    mapping = {
        "0-2": "beginner",
        "2-6": "beginner",
        "6-12": "intermediate1",
        "12-17": "intermediate1", 
        "17+": "advanced"
    }
    return mapping.get(training_time, "beginner")

@app.get("/evaluate-model")
async def evaluate_model(sample_size: int = 20):
    try:
        if workouts_df is None:
            raise HTTPException(status_code=503, detail="Dados não carregados")
        
        # Gerar dados de teste sintéticos baseados nos treinos existentes
        test_cases = []
        workout_ids = workouts_df['id'].unique()
        
        for _ in range(sample_size):
            workout = workouts_df.sample(1).iloc[0]
            test_cases.append({
                "gender": encoders['workoutType'].inverse_transform([workout['workoutType']])[0],
                "trainingTime": inverse_time_to_difficulty(workout['difficulty']),
                "goal": encoders['goal'].inverse_transform([workout['goal']])[0],
                "muscleFocus": encoders['category'].inverse_transform([workout['category']])[0],
                "weeklyFrequency": encoders['weeklyFrequency'].inverse_transform([workout['weeklyFrequency']])[0],
                "expected_workout_id": workout['id']
            })
        
        # Avaliar recomendações
        y_true = []
        y_pred = []
        
        for case in test_cases:
            try:
                customer_data = CustomerData(
                    name="Test User",
                    gender=case["gender"],
                    age=30,
                    weight=70,
                    goal=case["goal"],
                    trainingTime=case["trainingTime"],
                    weeklyFrequency=case["weeklyFrequency"],
                    muscleFocus=case["muscleFocus"]
                )
                
                response = await recommend_workout(customer_data)
                y_true.append(case["expected_workout_id"])
                y_pred.append(response["workout_id"])
            except Exception as e:
                print(f"Erro no caso de teste: {str(e)}")
                continue
        
        # Gerar matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=workout_ids)
        
        # Salvar visualização
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=workout_ids, yticklabels=workout_ids)
        plt.title("Matriz de Confusão - Sistema de Recomendação")
        plt.ylabel("Treino Esperado")
        plt.xlabel("Treino Recomendado")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Criar diretório se não existir
        Path("results").mkdir(exist_ok=True)
        plot_path = "results/confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        return {
            "status": "success",
            "plot_path": plot_path,
            "accuracy": np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def inverse_time_to_difficulty(difficulty_encoded: int) -> str:
    """Mapeia dificuldade codificada para tempo de treino aproximado"""
    difficulty = encoders['difficulty'].inverse_transform([difficulty_encoded])[0]
    mapping = {
        "beginner": "2-6",
        "intermediate1": "6-12",
        "advanced": "17+"
    }
    return mapping.get(difficulty, "2-6")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)