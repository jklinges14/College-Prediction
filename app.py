import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
#opening pickle file in read mode
clf = pickle.load(open('model.pkl', 'rb'))

#Home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #Get all the data from the imputs
    SAT = request.form.get('SAT')
    Subject_tests = request.form.get('Subject_tests')
    GPA = request.form.get('GPA')
    Rank = request.form.get('Rank')
    Gender = request.form.get('Gender')
    Ethnicity = request.form.get('Ethnicity')
    School_Type = request.form.get('School_Type')
    number_of_subject_tests = request.form.get('Number_of_subject_tests')

    new = [[SAT, Subject_tests, GPA, Rank, Gender, Ethnicity, School_Type, number_of_subject_tests]]
    final_features = pd.DataFrame(new, columns = ['SAT', 'Subject_tests', 'GPA', 'Rank', 'Gender', 'Ethnicity', 'School_Type', 'number_of_subject_tests'])
    prediction = clf.predict(final_features)

    output = prediction

    #Returning the results to the GUI
    if output == 1:
        return render_template('index.html', prediction_text='Congratulations! The model predicts that your application will be accepted by at least one of the top 15 universities.')

    if output == 0:
        return render_template('index.html', prediction_text='Unfortunately the model predicts that you would not be offered admission into a top 15 university')

if __name__ == "__main__":
    app.run(debug=True)
