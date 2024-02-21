import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Spam Detection using the Naive Bayes Classifier")
    st.write(
        """Replace with description of the dataset."""
        )
    
    data = pd.read_csv('spam.csv', 
                        dtype='str', header=0, 
                        sep = ",", encoding='latin')        
    st.subheader('The Dataset')
    # display the dataset
    st.header('The Dataset')
    
    st.dataframe(data, use_container_width=True)  
    X = data['v2']
    y = data['v1']        
    
    clfNB = make_pipeline(TfidfVectorizer(), MultinomialNB())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    clfNB.fit(X_train, y_train)
    
    y_test_pred = clfNB.predict(X_test)
    if st.button('Start'):
        
        st.header('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        st.write('Confusion Matrix')
        cmNB = confusion_matrix(y_test, y_test_pred)
        st.write(cmNB)

        text = 'receive a free entry'
        st.write(text + ' ---> ' + predict_category(clfNB, text))
        text = 'you could win a prize'
        st.write(text + ' ---> ' + predict_category(clfNB, text))
        text = 'We will have a meeting'
        st.write(text + ' ---> ' + predict_category(clfNB, text))
        text = 'camera for free'
        st.write(text + ' ---> ' + predict_category(clfNB, text))

    strinput = st.text_input("Enter message:")
    if st.button('Submit'):
        st.write(strinput + ' ---> ' + predict_category(clfNB, strinput))

def predict_category(clf, s):
    pred = clf.predict([s])
    return str(pred[0])

if __name__ == "__main__":
    app()
