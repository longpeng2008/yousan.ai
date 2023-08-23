#coding=utf8
from flask import url_for, redirect, render_template, jsonify, request, flash, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import Flask
import os
import time
import logging
import uuid
import codecs
import sys

## 载入功能模块，包括美学评分，表情识别等
sys.path.append('utils')
from utils.mouth.mouth import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__) ##标志该脚本为程序的根目录

app.config['SECRET_KEY'] = 'hard to guess'
app.config['CACHE'] = os.path.join(os.path.dirname(__file__), "static/cache")
app.config['MOUTH'] = os.path.join(os.path.dirname(__file__), "static/mouth")

# 表情识别图片所在目录
if not os.path.exists(app.config['MOUTH']):
	os.makedirs(app.config['MOUTH'])

# 返回给用户的图片所在目录
if not os.path.exists(app.config['CACHE']):
	os.makedirs(app.config['CACHE'])

## 创建路由映射，将index函数注册为路由，而且是根地址的处理程序
@app.route('/', methods=['GET'])
def index():
	return '欢迎来到有三AI小程序服务'

ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 表情识别
@app.route('/mouth', methods=['GET', 'POST'])
def get_mouth():
    file_data = request.files['file']
    if file_data and allowed_file(file_data.filename):
        filename = secure_filename(file_data.filename) ##获取名字
        file_uuid = str(uuid.uuid4().hex)
        time_now = datetime.now()
        filename = time_now.strftime("%Y%m%d%H%M%S")+"_"+file_uuid+"_"+filename
        file_data.save(os.path.join(app.config['MOUTH'], filename))
        src_path = os.path.join(app.config['MOUTH'], filename) ##获取服务端本地路径
        score = mouth(src_path) ##调用算法
        print("score is:"+str(score))
        emotion = 'None'
        if score == '0':
           emotion = "无表情"
        elif score == "1":
           emotion = "嘟嘴可爱"
        elif score == "2":
           emotion = "微笑"
        elif emotion == '3':
           emotion = "大笑"
        else:
           emotion = "无人脸"
	
        data = {
		"code": 0,
		"score": str(emotion)
	   }
        
        return jsonify(data)
    return jsonify({"code": 1, "msg": u"文件格式不允许"})

if __name__=='__main__':
    app.run(debug=True)
