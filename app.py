from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
@app.route('/saludo')
def saludo():
    return '¡Hola, mundo desde Colombia'
if __name__ == '__main__':
    app.run(debug = True, port = 4000)