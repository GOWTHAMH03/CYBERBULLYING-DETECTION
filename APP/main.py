# main.py
import os
import base64
import io
import math
from flask import Flask, flash, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
from datetime import datetime
from datetime import date
import random
from urllib.request import urlopen
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename
from PIL import Image

import urllib.request
import urllib.parse
import socket    
import csv
import xlrd 
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict

##
from textblob import TextBlob
#from nltk.corpus import wordnet
import re
#import nltk
#nltk.download()
from collections import Counter
#import plotly.express as px
#from nltk.stem import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer
#from gensim.utils import lemmatize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC ,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

plt.style.use('fivethirtyeight')
##

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="cyber_bullying"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s AND dstatus=0', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff=open("user.txt","w")
            ff.write(uname)
            ff.close()
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'

    return render_template('index.html',msg=msg)



@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM register")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    if request.method=='POST':
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        mobile=request.form['mobile']
        
        email=request.form['email']
        city=request.form['city']
        profession=request.form['profession']
        aadhar=request.form['aadhar']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        cursor = mydb.cursor()

        now = datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        sql = "INSERT INTO register(id,name,gender,dob,mobile,email,city,profession,aadhar,uname,pass,rdate) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,name,gender,dob,mobile,email,city,profession,aadhar,uname,pass1,rdate)
        cursor.execute(sql, val)
        mydb.commit()            
        print(cursor.rowcount, "Registered Success")
        result="sucess"
        if cursor.rowcount==1:
            return redirect(url_for('index'))
        else:
            msg='Already Exist'
    return render_template('/register.html',msg=msg)

@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    cnt=0
    uname=""
    
    
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()

    return render_template('userhome.html',msg=msg,data=data)

@app.route('/user_post', methods=['GET', 'POST'])
def user_post():
    st=0
    uname=""
    cnt=0
    act=""
    file_name=""
    if 'username' in session:
        uname = session['username']

    ff=open("user.txt","r")
    uname=ff.read()
    ff.close()
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE uname = %s', (uname, ))
    data = cursor.fetchone()

    pcursor = mydb.cursor()
    pcursor.execute('SELECT * FROM user_post u,register r where u.uname=r.uname && u.status=0 order by u.id desc')
    pdata = pcursor.fetchall()

    pcursor1 = mydb.cursor()
    pcursor1.execute('SELECT count(*) FROM user_post WHERE uname = %s and status=1', (uname, ))
    cnt = pcursor1.fetchone()[0]
    print(cnt)
    
    if request.method=='GET':
        act = request.args.get('act')
    if request.method == 'POST':
        post= request.form['message']
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM user_post")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        if file.filename == '':
            flash('No Select file')
            #return redirect(request.url)
        if file:
            fname = "P"+str(maxid)+file.filename
            file_name = secure_filename(fname)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER']+"/comments/", file_name))
            
        today = date.today()
        rdate = today.strftime("%d-%m-%Y")

        cursor2 = mydb.cursor()
        
        ########
        '''loc = ("dataset.xlsx") 
        # To open Workbook 
        wb = xlrd.open_workbook(loc) 
        sheet = wb.sheet_by_index(0)
        nr=sheet.nrows
        i=0
        while i<nr:
            #print(sheet.cell_value(i, 0))
            dd=sheet.cell_value(i, 0)
            if post.find(dd) != -1:
                
                break
            i+=1'''
        ###########
        x=0
        deptype=0
        f1=open("static/test.txt","r")
        dat=f1.read()
        f1.close()
        dat1=dat.split("|")
        for rd in dat1:
           
            t1=post
            t2=rd.strip()
            if t2 in t1:
                act="yes"
                st=1
                x+=1
                break
        ###########
        if cnt==2:
            act="warn"
        elif cnt>=3:
            
            pcursor2 = mydb.cursor()
            pcursor2.execute('update register set dstatus=1 where uname = %s', (uname, ))
            mydb.commit()
            act="block"

        
        sql = "INSERT INTO user_post (id,uname,text_post,photo,rdate,status) VALUES(%s,%s,%s,%s,%s,%s)"
        val = (maxid,uname,post,file_name,rdate,st)
        mycursor.execute(sql,val)
        print(sql,val)
        mydb.commit()
        msg="Upload success"
        return redirect(url_for('user_post',act=act))  
    
    return render_template('user_post.html',data=data,act=act,pdata=pdata)



@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    uname=""
    if 'username' in session:
        uname = session['username']
        print(uname)    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM register WHERE uname = %s', (uname, ))
    data = mycursor.fetchone()
    
    
    if request.method=='POST':
        name = request.form['name']
        dob = request.form['dob']
        contact = request.form['mobile']
        email = request.form['email']
        location = request.form['location']
        profession = request.form['profession']
        aadhar = request.form['aadhar']
        #filename=('uname.txt')
        #fileread=open(filename,"r+")
        #uname=fileread.read()
        #fileread.close()
        
        sql=("update register set name=%s, dob=%s,mobile=%s,email=%s,location=%s,profession=%s,aadhar=%s,status=1 where uname=%s")
        val=(name,dob, contact, email, location, profession,aadhar, uname)
        mycursor.execute(sql,val)
        mydb.commit()
        print(val)
        msg="success"
        return redirect(url_for('userhome',msg=msg))
    return render_template('edit_profile.html',data=data)


@app.route('/change_profile', methods=['GET', 'POST'])
def change_profile():
    uid=""
    uname=""
    print(uid)
    if 'username' in session:
        uname = session['username']
    print(uname)

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()
    
    if request.method=='GET':
        act = request.args.get('act')
        uid = request.args.get('uname')
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']
        print(file)
        if file.filename == '':
            flash('No Select file')
            return redirect(request.url)
        if file:
            fname = file.filename
            fimg = uname+".png"
            file_name = secure_filename(fimg)
            print(file_name)
            file.save(os.path.join(app.config['UPLOAD_FOLDER']+"/photo/", file_name))
            
            
            mycursor.execute("update register set photo=1 where uname=%s", (uname, ))
            mydb.commit()
            msg="Upload success"
            return redirect(url_for('userhome'))  
    
    return render_template('change_profile.html',data=data)

@app.route('/admin_user_view', methods=['GET', 'POST'])
def admin_user_view():
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register where dstatus=0')
    data = cursor.fetchall()
    return render_template('admin_user_view.html',data=data)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register where dstatus=0')
    data = cursor.fetchall()
    cursor.execute('SELECT * FROM register where dstatus=1')
    data2 = cursor.fetchall()

    cursor.execute('SELECT count(*) FROM user_post')
    cnt = cursor.fetchone()[0]

    cursor.execute('SELECT count(*) FROM user_post where status=0')
    cnt2 = cursor.fetchone()[0]

    cursor.execute('SELECT count(*) FROM user_post where status=1')
    cnt3 = cursor.fetchone()[0]

    per_hu=0
    per_bot=0

    if cnt2>0:
        per_hu=(cnt2/cnt)*100
    else:
        per_hu=0
        
    if cnt3>0:
        per_bot=(cnt3/cnt)*100
    else:
        per_bot=0
    
    dat=['Not Cyberbullying','Cyberbullying']
    dat1=[per_hu,per_bot]
    courses = dat #list(data.keys())
    values = dat1 #list(data.values())
      
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(courses, values, color ='maroon',
            width = 0.4)
 


    plt.xlabel("Prediction")
    plt.ylabel("Percentage")
    plt.title("")

    
    fn="result.png"
    plt.savefig('static/chart/'+fn)
    #plt.close()
    plt.clf()
    
    return render_template('prediction.html',data=data,data2=data2,fn=fn)



############################################
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""

    #get the data
    df =  pd.read_csv('dataset/cyberbullying_tweets.csv')
    data=[]
    #show the top 5 data
    dat=df.head()

    data=[]
    for ss in dat.values:
        data.append(ss)

    #df.info()
        
    
    return render_template('admin.html',msg=msg,data=data)

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""

    df =  pd.read_csv('dataset/cyberbullying_tweets.csv')

    #stopwords=STOPWORDS
    stemmer = PorterStemmer()
    #stemmer = SnowballStemmer("english")
    #lematizer=WordNetLemmatizer()

    from wordcloud import STOPWORDS
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                      'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                      'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                      'de', 're', 'amp', 'will'])

    def lower(text):
        return text.lower()

    def remove_hashtag(text):
        return re.sub("#[A-Za-z0-9_]+", ' ', text)

    def remove_twitter(text):
        return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

    def remove_stopwords(text):
        return " ".join([word for word in 
                         str(text).split() if word not in STOPWORDS])

    def stemming(text):
        return " ".join([stemmer.stem(word) for word in text.split()])

    #def lemmatizer_words(text):
    #    return " ".join([lematizer.lemmatize(word) for word in text.split()])

    def cleanTxt(text):
        text = lower(text)
        text = remove_hashtag(text)
        text = remove_twitter(text)
        text = remove_stopwords(text)
        text = stemming(text)
        #text = lemmatizer_words(text)
        return text

    #cleaning the text
    df['tweet_clean'] = df['tweet_text'].apply(cleanTxt)

    #show the clean text
    dat=df.head()
    data=[]
    for ss in dat.values:
        data.append(ss)

    
    return render_template('process1.html',data=data)

######
@app.route('/process11', methods=['GET', 'POST'])
def process11():
    msg=""
    cnt=0
    rows=0
    cols=0
    data=[]

    df =  pd.read_csv('dataset/cyberbullying_tweets.csv')
    
    #stemmer = SnowballStemmer("english")
    #lematizer=WordNetLemmatizer()
    #stopwords=STOPWORDS
    stemmer = PorterStemmer()
    
    from wordcloud import STOPWORDS
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                      'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                      'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                      'de', 're', 'amp', 'will'])

    def lower(text):
        return text.lower()

    def remove_hashtag(text):
        return re.sub("#[A-Za-z0-9_]+", ' ', text)

    def remove_twitter(text):
        return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

    def remove_stopwords(text):
        return " ".join([word for word in 
                         str(text).split() if word not in STOPWORDS])

    def stemming(text):
        return " ".join([stemmer.stem(word) for word in text.split()])

    #def lemmatizer_words(text):
    #    return " ".join([lematizer.lemmatize(word) for word in text.split()])

    def cleanTxt(text):
        text = lower(text)
        text = remove_hashtag(text)
        text = remove_twitter(text)
        text = remove_stopwords(text)
        text = stemming(text)
        #text = lemmatizer_words(text)
        return text

    #cleaning the text
    df['tweet_clean'] = df['tweet_text'].apply(cleanTxt)

    #show the clean text
    dat=df.head()

    #data visulization
    df['tweet_list'] = df['tweet_clean'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in df['tweet_list'] for item in sublist])
    tweet_list1 = pd.DataFrame(top.most_common(20))
    tweet_list1.columns = ['Words','Count']
    tweet_list1.style.background_gradient(cmap='Greens')


    for ss in tweet_list1.values:
        
        data.append(ss)

    ##graph1
    #figure = px.bar(tweet_list1, x="Count", y="Words", title='Top 20 words in cyberbullying tweet', orientation='h', width=700, height=700,color='Words')
    #plt.savefig("static/graph/graph1.png")
    #figure.show()

    ##
    not_cyberbullying_type = df[df['cyberbullying_type']=='not_cyberbullying']
    gender_type = df[df['cyberbullying_type']=='gender']
    religion_type = df[df['cyberbullying_type']=='religion']
    other_cyberbullying_type = df[df['cyberbullying_type']=='other_cyberbullying']
    age_type = df[df['cyberbullying_type']=='age']
    ethnicity_type = df[df['cyberbullying_type']=='ethnicity']

    #Top 20 Words in not cyberbullying Tweet
    top20 = Counter([item for sublist in not_cyberbullying_type['tweet_list'] for item in sublist])
    type_nc = pd.DataFrame(top20.most_common(20))
    type_nc.columns = ['Top of Words','Count']
    type_nc.style.background_gradient(cmap='Greens')

    data2=[]
    for ss2 in type_nc.values:
        
        data2.append(ss2)

    #graph2
    #nc_fig = px.bar(type_nc, x="Count", y="Top of Words", title='Top 20 Words in not cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #nc_fig.show()

    ########################
    #Top 20 Words in Gender cyberbullying Tweet
    top20_gender = Counter([item for sublist in gender_type['tweet_list'] for item in sublist])
    type_g = pd.DataFrame(top20_gender.most_common(20))
    type_g.columns = ['Top of Words','Count']
    type_g.style.background_gradient(cmap='Greens')

    data3=[]
    for ss3 in type_g.values:
        
        data3.append(ss3)
    #g3
    #g_fig = px.bar(type_g, x="Count", y="Top of Words", title='Top 20 Words in Gender Cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #g_fig.show()
    ##########

    #Top 20 Words in religion cyberbullying Tweet
    top20_r = Counter([item for sublist in religion_type['tweet_list'] for item in sublist])
    type_r = pd.DataFrame(top20_r.most_common(20))
    type_r.columns = ['Top of Words','Count']
    type_r.style.background_gradient(cmap='Greens')

    data4=[]
    for ss4 in type_r.values:
        
        data4.append(ss4)
    #g4
    #r_fig = px.bar(type_r, x="Count", y="Top of Words", title='Top 20 Words in Religion Cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #r_fig.show()

    #######
    #Top 20 Words in others cyberbullying Tweet
    top20_o = Counter([item for sublist in other_cyberbullying_type['tweet_list'] for item in sublist])
    type_o = pd.DataFrame(top20_o.most_common(20))
    type_o.columns = ['Top of Words','Count']
    type_o.style.background_gradient(cmap='Greens')

    data5=[]
    for ss5 in type_o.values:
        
        data5.append(ss5)

    #o_fig = px.bar(type_o, x="Count", y="Top of Words", title='Top 20 Words in Other Cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #o_fig.show()
    ############

    #Top 20 Words in Age Cyberbullying Tweet
    top20_a = Counter([item for sublist in age_type['tweet_list'] for item in sublist])
    type_a = pd.DataFrame(top20_a.most_common(20))
    type_a.columns = ['Top of Words','Count']
    type_a.style.background_gradient(cmap='Greens')
    data6=[]
    for ss6 in type_a.values:
        
        data6.append(ss6)

    #g6
    #a_fig = px.bar(type_a, x="Count", y="Top of Words", title='Top 20 Words in Age cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #a_fig.show()

    ########
    #Top 20 Words in Ethnicity Cyberbullying Tweet
    top20_e = Counter([item for sublist in ethnicity_type['tweet_list'] for item in sublist])
    type_e = pd.DataFrame(top20_e.most_common(20))
    type_e.columns = ['Top of Words','Count']
    type_e.style.background_gradient(cmap='Greens')
    data7=[]
    for ss7 in type_e.values:
        
        data7.append(ss7)
    ##g7
    #e_fig = px.bar(type_e, x="Count", y="Top of Words", title='Top 20 Words in Ethnicity Cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #e_fig.show()
    ###############
    #spliting the data
    labels = df['cyberbullying_type'].tolist()
    df.cyberbullying_type.unique()
    ClassIDMap = {'not_cyberbullying': 1, 'gender':2, 
              'religion':3, 'other_cyberbullying': 4, 
              'age': 5, 'ethnicity': 6 }
    ClassIDMap
    corpus, target_labels, target_names = (df['tweet_clean'], 
                                       [ClassIDMap[label] for 
                                        label in df['cyberbullying_type']], 
                                       df['cyberbullying_type'])

    df = pd.DataFrame({'tweet text': corpus, 'cyberbullying Label': 
                        target_labels, 'cyberbulying Name': target_names})

    #df.info()
    ########################################
    #Spliting data
    train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names =\
                                 train_test_split(np.array(df['tweet text']), np.array(df['cyberbullying Label']),
                                                       np.array(df['cyberbulying Name']), test_size=0.33, random_state=42)

    train=train_corpus.shape
    test=test_corpus.shape

    ################
    

    dt_clean=[]
    df1 =  pd.read_csv('dataset/data.csv')
    n=0
    i=1
    for ss8 in df1.values:
        if i<=20:
            print(ss8[1])
            #if pd.isnull(ss8[1]):
            #    n+=1
            #else:
            dt_clean.append(ss8)
            

        i+=1
        

    
    
    return render_template('process11.html',dt_clean=dt_clean,train=train,test=test)



@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    cnt=0
    rows=0
    cols=0
    data=[]

    df =  pd.read_csv('dataset/cyberbullying_tweets.csv')
    
    #stemmer = SnowballStemmer("english")
    #lematizer=WordNetLemmatizer()
    #stopwords=STOPWORDS
    stemmer = PorterStemmer()
    
    from wordcloud import STOPWORDS
    STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                      'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                      'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                      'de', 're', 'amp', 'will'])

    def lower(text):
        return text.lower()

    def remove_hashtag(text):
        return re.sub("#[A-Za-z0-9_]+", ' ', text)

    def remove_twitter(text):
        return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

    def remove_stopwords(text):
        return " ".join([word for word in 
                         str(text).split() if word not in STOPWORDS])

    def stemming(text):
        return " ".join([stemmer.stem(word) for word in text.split()])

    #def lemmatizer_words(text):
    #    return " ".join([lematizer.lemmatize(word) for word in text.split()])

    def cleanTxt(text):
        text = lower(text)
        text = remove_hashtag(text)
        text = remove_twitter(text)
        text = remove_stopwords(text)
        text = stemming(text)
        #text = lemmatizer_words(text)
        return text

    #cleaning the text
    df['tweet_clean'] = df['tweet_text'].apply(cleanTxt)

    #show the clean text
    dat=df.head()

    #data visulization
    df['tweet_list'] = df['tweet_clean'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in df['tweet_list'] for item in sublist])
    tweet_list1 = pd.DataFrame(top.most_common(20))
    tweet_list1.columns = ['Words','Count']
    tweet_list1.style.background_gradient(cmap='Greens')


    for ss in tweet_list1.values:
        
        data.append(ss)

    ##graph1
    #figure = px.bar(tweet_list1, x="Count", y="Words", title='Top 20 words in cyberbullying tweet', orientation='h', width=700, height=700,color='Words')
    #plt.savefig("static/graph/graph1.png")
    #figure.show()

    ##
    not_cyberbullying_type = df[df['cyberbullying_type']=='not_cyberbullying']
    gender_type = df[df['cyberbullying_type']=='gender']
    religion_type = df[df['cyberbullying_type']=='religion']
    other_cyberbullying_type = df[df['cyberbullying_type']=='other_cyberbullying']
    age_type = df[df['cyberbullying_type']=='age']
    ethnicity_type = df[df['cyberbullying_type']=='ethnicity']

    #Top 20 Words in not cyberbullying Tweet
    top20 = Counter([item for sublist in not_cyberbullying_type['tweet_list'] for item in sublist])
    type_nc = pd.DataFrame(top20.most_common(20))
    type_nc.columns = ['Top of Words','Count']
    type_nc.style.background_gradient(cmap='Greens')

    data2=[]
    for ss2 in type_nc.values:
        
        data2.append(ss2)

    #graph2
    #nc_fig = px.bar(type_nc, x="Count", y="Top of Words", title='Top 20 Words in not cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #nc_fig.show()

    ########################
    #Top 20 Words in Gender cyberbullying Tweet
    top20_gender = Counter([item for sublist in gender_type['tweet_list'] for item in sublist])
    type_g = pd.DataFrame(top20_gender.most_common(20))
    type_g.columns = ['Top of Words','Count']
    type_g.style.background_gradient(cmap='Greens')

    data3=[]
    for ss3 in type_g.values:
        
        data3.append(ss3)
    #g3
    #g_fig = px.bar(type_g, x="Count", y="Top of Words", title='Top 20 Words in Gender Cyberbullying Tweet', orientation='h', 
    #        width=700, height=700,color='Top of Words')
    #g_fig.show()
    ##########

    #Top 20 Words in religion cyberbullying Tweet
    top20_r = Counter([item for sublist in religion_type['tweet_list'] for item in sublist])
    type_r = pd.DataFrame(top20_r.most_common(20))
    type_r.columns = ['Top of Words','Count']
    type_r.style.background_gradient(cmap='Greens')

    data4=[]
    for ss4 in type_r.values:
        
        data4.append(ss4)
    #g4
    #r_fig = px.bar(type_r, x="Count", y="Top of Words", title='Top 20 Words in Religion Cyberbullying Tweet', orientation='h', 
    #        width=700, height=700,color='Top of Words')
    #r_fig.show()

    #######
    #Top 20 Words in others cyberbullying Tweet
    top20_o = Counter([item for sublist in other_cyberbullying_type['tweet_list'] for item in sublist])
    type_o = pd.DataFrame(top20_o.most_common(20))
    type_o.columns = ['Top of Words','Count']
    type_o.style.background_gradient(cmap='Greens')

    data5=[]
    for ss5 in type_o.values:
        
        data5.append(ss5)

    #o_fig = px.bar(type_o, x="Count", y="Top of Words", title='Top 20 Words in Other Cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #o_fig.show()
    ############

    #Top 20 Words in Age Cyberbullying Tweet
    top20_a = Counter([item for sublist in age_type['tweet_list'] for item in sublist])
    type_a = pd.DataFrame(top20_a.most_common(20))
    type_a.columns = ['Top of Words','Count']
    type_a.style.background_gradient(cmap='Greens')
    data6=[]
    for ss6 in type_a.values:
        
        data6.append(ss6)

    #g6
    #a_fig = px.bar(type_a, x="Count", y="Top of Words", title='Top 20 Words in Age cyberbullying Tweet', orientation='h', 
    #         width=700, height=700,color='Top of Words')
    #a_fig.show()

    ########
    #Top 20 Words in Ethnicity Cyberbullying Tweet
    top20_e = Counter([item for sublist in ethnicity_type['tweet_list'] for item in sublist])
    type_e = pd.DataFrame(top20_e.most_common(20))
    type_e.columns = ['Top of Words','Count']
    type_e.style.background_gradient(cmap='Greens')
    data7=[]
    for ss7 in type_e.values:
        
        data7.append(ss7)
    ##g7
    #e_fig = px.bar(type_e, x="Count", y="Top of Words", title='Top 20 Words in Ethnicity Cyberbullying Tweet', orientation='h', 
    #        width=700, height=700,color='Top of Words')
    #e_fig.show()
    ###############
    #spliting the data
    labels = df['cyberbullying_type'].tolist()
    df.cyberbullying_type.unique()
    ClassIDMap = {'not_cyberbullying': 1, 'gender':2, 
              'religion':3, 'other_cyberbullying': 4, 
              'age': 5, 'ethnicity': 6 }
    ClassIDMap
    corpus, target_labels, target_names = (df['tweet_clean'], 
                                       [ClassIDMap[label] for 
                                        label in df['cyberbullying_type']], 
                                       df['cyberbullying_type'])

    df = pd.DataFrame({'tweet text': corpus, 'cyberbullying Label': 
                        target_labels, 'cyberbulying Name': target_names})

    
    data8=[]
    i=0
    col=0
    for ss8 in df.values:
        col=len(ss8)
        data8.append(ss8)
        i+=1
    row=i
    
    #df.info()

  
    
    return render_template('process2.html',data=data,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7,data8=data8,row=row,col=col)


#LE-LSTM Training Loop
def train_data(train_loader):
    total_step = len(train_loader)
    total_step_val = len(valid_loader)

    early_stopping_patience = 4
    early_stopping_counter = 0

    valid_acc_max = 0 # Initialize best accuracy top 0

    for e in range(EPOCHS):

        #lists to host the train and validation losses of every batch for each epoch
        train_loss, valid_loss  = [], []
        #lists to host the train and validation accuracy of every batch for each epoch
        train_acc, valid_acc  = [], []

        #lists to host the train and validation predictions of every batch for each epoch
        y_train_list, y_val_list = [], []

        #initalize number of total and correctly classified texts during training and validation
        correct, correct_val = 0, 0
        total, total_val = 0, 0
        running_loss, running_loss_val = 0, 0


        ####TRAINING LOOP####

        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) #load features and targets in device

            h = model.init_hidden(labels.size(0))

            model.zero_grad() #reset gradients 

            output, h = model(inputs,h) #get output and hidden states from LSTM network
            
            loss = criterion(output, labels)
            loss.backward()
            
            running_loss += loss.item()
            
            optimizer.step()

            y_pred_train = torch.argmax(output, dim=1) #get tensor of predicted values on the training set
            y_train_list.extend(y_pred_train.squeeze().tolist()) #transform tensor to list and the values to the list
            
            correct += torch.sum(y_pred_train==labels).item() #count correctly classified texts per batch
            total += labels.size(0) #count total texts per batch

        train_loss.append(running_loss / total_step)
        train_acc.append(100 * correct / total)

        ####VALIDATION LOOP####
        
        with torch.no_grad():
            
            model.eval()
            
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                val_h = model.init_hidden(labels.size(0))

                output, val_h = model(inputs, val_h)

                val_loss = criterion(output, labels)
                running_loss_val += val_loss.item()

                y_pred_val = torch.argmax(output, dim=1)
                y_val_list.extend(y_pred_val.squeeze().tolist())

                correct_val += torch.sum(y_pred_val==labels).item()
                total_val += labels.size(0)

            valid_loss.append(running_loss_val / total_step_val)
            valid_acc.append(100 * correct_val / total_val)

        #Save model if validation accuracy increases
        if np.mean(valid_acc) >= valid_acc_max:
            torch.save(model.state_dict(), './state_dict.pt')
            print(f'Epoch {e+1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ...')
            valid_acc_max = np.mean(valid_acc)
            early_stopping_counter=0 #reset counter if validation accuracy increases
        else:
            print(f'Epoch {e+1}:Validation accuracy did not increase')
            early_stopping_counter+=1 #increase counter if validation accuracy does not increase
            
        if early_stopping_counter > early_stopping_patience:
            print('Early stopped at epoch :', e+1)
            break
        
        print(f'\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}')
        print(f'\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%')

#LE-LSTM-Testing
def test_data():
    model.eval()
    y_pred_list = []
    y_test_list = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        test_h = model.init_hidden(labels.size(0))

        output, val_h = model(inputs, test_h)
        y_pred_test = torch.argmax(output, dim=1)
        y_pred_list.extend(y_pred_test.squeeze().tolist())
        y_test_list.extend(labels.squeeze().tolist())
##

#####################
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    ###
    f1=open("static/det.txt","r")
    read1=f1.read()
    f1.close()
    rdata=read1.split(',')
    v1=int(rdata[0])
    v2=int(rdata[1])
    tot=v1+v2
    dd2=[v1,v2]
    dd1=['Cyberbullying','Not Cyberbullying']
    
    doc = dd1 #list(data.keys())
    values = dd2 #list(data.values())

    f11=open("static/det1.txt","r")
    rk=f11.read()
    f11.close()
    rk1=rk.split('|')
    f12=open("static/det11.txt","r")
    pk=f12.read()
    f12.close()
    i=0
    pk1=pk.split('|')
    print(pk1)
    data=[]
    while i<6:
        pk2=[]
        pkk=pk1[i].split('-')
        pk2.append(pkk[0])
        pk2.append(pkk[1])
        pk2.append(pkk[2])
        pk2.append(pkk[3])
        pk2.append(pkk[4])
        
        data.append(pk2)
        i+=1
    data1=pk1[6].split('-')
    data2=pk1[7].split('-')
    data3=pk1[8].split('-')
    fig = plt.figure(figsize = (12, 8))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.2)
 

    plt.ylim((1,tot))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")
    
    fn="classify.png"
    plt.xticks(rotation=0)
    plt.savefig('static/graph/'+fn)
    
    #plt.close()
    plt.clf()
    #########
    #
    y=[]
    x1=[]
    x2=[]

    
    
    x1=rk1[0].split(',')
    y=rk1[4].split(',')
    x2=rk1[1].split(',')
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    
    dd=["Training","Validation"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("")

    
    fn="acc1.png"
    
    #plt.savefig('static/graph/'+fn)
    #plt.close()
    plt.clf()
    ####
    y=[]
    x1=[]
    x2=[]

    
    
    x1=rk1[2].split(',')
    y=rk1[4].split(',')
    x2=rk1[3].split(',')
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["Training","Validation"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("")

    
    fn="acc2.png"
    
    #plt.savefig('static/graph/'+fn)
    #plt.close()
    plt.clf()
    ##################
    

    return render_template('classify.html',v1=v1,v2=v2,data=data,data1=data1,data2=data2,data3=data3)

##########################







@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)


