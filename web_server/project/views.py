from . import app
from flask import render_template
from flask import request
import os



@app.route('/')
def index():
    return "welmoe to heuristic lab"

UPLOAD_FOLDER = '~/Documents/MLProject/stackmint/web_server/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/stackmint', methods=['POST'])
def stackmint():
    file = request.files['photo']
    file.save(file.filename)
    # file.save(os.path.join("stackMint",file.filename))

    locationX = request.form.get('locationX')
    locationY = request.form.get('locationY')
    width = request.form.get('width')
    height = request.form.get('height')


    print(locationX, locationY, width, height)
    return "success"
