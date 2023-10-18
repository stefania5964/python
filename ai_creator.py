import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import csv
import json
from tensorflow.keras.models import load_model

def getData(): 
	json_file = open('../premierwithplayers.json')
	data = json.load(json_file)
	home_ids = []
	away_ids = []
	victory = []
	avg_home_score = []
	avg_away_score = []
	
	for match in data:
		home_ids.append(int(match["match_hometeam_id"]))
		away_ids.append(int(match["match_awayteam_id"]))
		victory.append(1 if match["match_hometeam_score"] == match["match_awayteam_score"] else (0 if match["match_hometeam_score"] > match["match_awayteam_score"] else 2))		
	
		home_score = 0.0
		home_count = 0
		away_score = 0.0
		away_count = 0
		
		for player in match["player_stats"]["home"]:
			if player["player_rating"] == "0":
				continue
				
			home_score += float(player["player_rating"])
			home_count += 1
			
		for player in match["player_stats"]["away"]:
			if player["player_rating"] == "0":
				continue
				
			away_score += float(player["player_rating"])
			away_count += 1
	
		avg_home_score.append(home_score / home_count)
		avg_away_score.append(away_score / away_count)
	
	return (home_ids, away_ids, victory, avg_home_score, avg_away_score)

def generateAI(home_ids, away_ids, victory, avg_home_score, avg_away_score):
	# Supongamos que tienes datos en forma de arrays o dataframes.
	# Los equipos ya deben estar codificados en números enteros únicos.
	
	# Datos de entrada
	equipo_local = home_ids  # Códigos de equipos locales
	equipo_visitante = away_ids  # Códigos de equipos visitantes
	porcentaje_victorias_local = [0.60, 0.65, 0.70, 0.55, ...]  # Porcentaje de victorias del equipo local
	porcentaje_victorias_visitante = [0.70, 0.75, 0.65, 0.60, ...]  # Porcentaje de victorias del equipo visitante
	probabilidad_ganar_local = victory  # Probabilidad de que el equipo local gane
	
	# Dividir los datos en conjuntos de entrenamiento y prueba
	X = np.column_stack((avg_home_score,avg_away_score))#, porcentaje_victorias_local, porcentaje_victorias_visitante))
	y = np.array(probabilidad_ganar_local)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	y_train = to_categorical(y_train, num_classes=3)
	y_test = to_categorical(y_test, num_classes=3)
	
	# Crear un modelo de red neuronal
	model = keras.Sequential([
    	keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    	keras.layers.Dropout(0.1),
    	keras.layers.Dense(32, activation='relu'),
    	keras.layers.Dense(3, activation='softmax')  # Usar 'linear' para regresión
	])
	
	# Crear una instancia de EarlyStopping
	early_stopping = EarlyStopping(monitor='accuracy',  # Monitorea la pérdida en el conjunto de validación
    patience=10,          # Número de épocas sin mejora antes de detener
    restore_best_weights=False)  # Restaurar los mejores pesos del modelo

	# Entrenar el modelo con EarlyStopping
	
	# Compilar el modelo
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	
	# Entrenar el modelo
	model.fit(X_train, y_train, epochs=100, batch_size=4,callbacks=[early_stopping])
	
	# Evaluar el modelo en los datos de prueba
	loss = model.evaluate(X_test, y_test)
	print(f'Pérdida en datos de prueba: {loss}')
	
	# Realizar predicciones
	predicciones = model.predict(X_test, verbose=2)
	
	for x_value, y_value in zip(X_test, y_test):
		print(f"Valor en X: {x_value}, Resultado en Y: {y_value}")
		
	model.save("modelo1.keras")
    
	
(home_ids, away_ids, victory, avg_home_score, avg_away_score) = getData()
generateAI(home_ids, away_ids, victory, avg_home_score, avg_away_score)	
