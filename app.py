from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')
    return 'Hello World'


@app.route('/initial_inputs')
def initial_input():
    return render_template('initial_inputs.html')


if __name__ == '__main__':
    app.run(debug=True)
