#Import flas
from flask import Flask


#Config
app = Flask(__name__)
from . import views