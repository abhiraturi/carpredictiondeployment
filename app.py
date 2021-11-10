#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import numpy as np
import flask
import pickle
from flask import Flask,redirect,url_for,request,render_template


# In[7]:


#creating instance of the class
app=Flask(__name__,template_folder='templates')

#to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# In[8]:


#prediction function
def ValuePredictor(to_predict_list):
    to_predict=np.array(to_predict_list).reshape(1,7)
    loaded_model=pickle.load(open("model.pkl","rb"))
    result=loaded_model.predict(to_predict)
    return(result[0])

# @app.route('/result',methods=['POST'])
# def result():
#     prediction=result
#     return render_template("result.html",prediction=prediction)

@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        #prediction=result
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = round(ValuePredictor(to_predict_list),2)              
        return render_template("result.html", prediction = result)


# In[9]:


# if __name__=="main":
#     app.run(debug=True,port=8000)


# In[10]:


try: 
    # works fine while running the py script in the command-line 
    app.run(debug=True,port=9000)
except:
    # internal issue with Jupyter Notebook
    print("Exception occured!")
    # running manually
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)


# In[ ]:




