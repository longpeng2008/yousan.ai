#coding:utf8
from flask import Flask 
app = Flask(__name__) 

@app.route("/") 
def hello(): 
    return "欢迎来到有三AI"
 
if __name__ == "__main__":
    app.run()

