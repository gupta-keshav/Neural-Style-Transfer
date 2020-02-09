from flask import Flask,render_template,redirect,request
import numpy as np
import os
from styletransfer import create_result

app = Flask(__name__)


@app.route('/')
def hello():
	return render_template("index.html")

								#submit button #url	
@app.route('/about',methods = ['POST','GET'])
def about():
	if request.method == 'POST':
		
		for file in os.listdir('./static/'):
    			os.remove('./static/'+file)
				
		f = request.files['img']
		path = "./static/{}".format(f.filename)
		f.save(path)

		f1 = request.files['img1']
		path1 = "./static/{}".format(f1.filename)
		f1.save(path1)
		result_path = './static/{}-{}.jpg'.format(f.filename, f1.filename)
		create_result(path, path1, result_path)

		image_path = {
		'a': path,
		'b': path1,
		'c': result_path
		}
		return render_template("index.html",result = image_path)
	else:
		return "HEHE"

@app.route('/home')
def home():
	return redirect('/')

if __name__ == '__main__':
	   
	app.run(debug = True )








