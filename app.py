from fileinput import filename
import flask
import numpy as np
from flask import Flask ,request,jsonify,render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB


filename = 'model.pkl'
clf=pickle.load(open(filename,'rb'))
cv=pickle.load(open('transform.pkl','rb'))
ps=pickle.load(open('portstem.pkl','rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/CHECK',methods = ['GET','POST'])
def CHECK():
    if request.method == 'POST':
        message = request.form['message']
        message1=np.array([message])
        data = []
       
        for i in range (0,len(message1)):
            review = re.sub('[^a-zA-Z]',' ',message1[i])
            review = review.lower()
            review = review.split()
            review = [ps.stem(word)for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            data.append(review)
            print(review)
        print (message)
       
        print(data)
        vect = cv.transform(data).toarray()
        print (vect)
        my_prediction = clf.predict(vect)
        
        print ([my_prediction])
    return render_template('result.html',prediction = my_prediction)
    

if __name__=='__main__':
    app.run(debug=True)
        
        