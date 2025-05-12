from firebase_admin import firestore

def load_historical_data_from_firebase():
    """Carrega dados históricos de clientes e seus treinos do Firebase"""
    db = firestore.client()
    
    # 1. Buscar todos os clientes que têm currentWorkoutId
    customers_ref = db.collection('customers')
    customers = [doc.to_dict() for doc in customers_ref.stream() if 'currentWorkoutId' in doc.to_dict()]
    
    historical_data = []
    
    for customer in customers:
        # 2. Buscar o treino correspondente
        if customer['currentWorkoutId']:
            workout_ref = db.collection('workouts').document(customer['currentWorkoutId'])
            workout = workout_ref.get()
            
            if workout.exists:
                # 3. Criar par (dados_do_cliente, workout_id)
                historical_data.append((customer, customer['currentWorkoutId']))
    
    return historical_data