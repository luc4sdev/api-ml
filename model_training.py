from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_advanced_model():
    # 1. Carregar dados históricos de usuários e seus treinos
    # (Você precisará criar esse dataset com pares usuário-treino)
    historical_data = load_historical_data_from_firebase()
    
    # 2. Preparar features (X) e labels (y)
    X = []
    y = []
    
    for user_data, workout_id in historical_data:
        features = preprocess_client(user_data)
        X.append([
            features['gender'],
            features['age'],
            features['weight'],
            features['goal'],
            features['trainingTime'],
            features['weeklyFrequency'],
            features['muscleFocus']
        ])
        y.append(workout_id)
    
    # 3. Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 4. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
    
    # 5. Treinar modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 6. Avaliar
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model, label_encoder

# Você pode salvar o modelo treinado para uso posterior
import joblib
model, encoder = train_advanced_model()
joblib.dump(model, 'workout_recommender_model.joblib')
joblib.dump(encoder, 'label_encoder.joblib')