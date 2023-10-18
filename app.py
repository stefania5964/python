from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

# Obtiene el directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambia el directorio de trabajo al directorio del script
os.chdir(script_dir)

app = Flask(__name__)
model = load_model("modelo1.keras")
json_file = open('premierwithplayers.json')
data = json.load(json_file)
teams = {}

for match in data:
	home_score = 0
	home_count = 0
	
	for player in match["player_stats"]["home"]:
		if player["player_rating"] == "0":
			continue
			
		home_score += float(player["player_rating"])
		home_count += 1
	
	score = home_score / home_count

	teams[match["match_hometeam_name"]] = {
		'id': match["match_hometeam_id"],
		'average_score': score
	}

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/saludo')
def saludo():
    return '¡Hola, mundo desde Colombia'
    
@app.route('/predict')
def predict():
	equipo1 = teams[request.args.get('home')]
	equipo2 = teams[request.args.get('away')]
	
	entrada = np.array([[equipo1["average_score"], equipo2["average_score"]]])
	
	prediccion = model.predict(entrada)
	etiqueta_predicha = np.argmax(prediccion)
	
	resultado = {
		'resultado': float(etiqueta_predicha),
		'probabilidad': float(prediccion[0][etiqueta_predicha])
	}
	
	return jsonify(resultado)
	
    
if __name__ == '__main__':
    app.run(debug = True, port = 4000)
