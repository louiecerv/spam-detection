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
    if st.button('Start'):
        data = pd.read_csv('spam.csv', 
                           dtype='str', header=0, 
                           sep = ",", encoding='latin')        
        X = data['v2']
        y = data['v1']        
        
        clfNB = make_pipeline(TfidfVectorizer(), MultinomialNB())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        clfNB.fit(X_train, y_train)
        X_test_pred = clfNB.predict(X_test)
        accuracy = 100.0 * (y_test == X_test_pred).sum() / X_test.shape[0]
        print("Accuracy of the NB classifier =", round(accuracy, 2), "%")
        cmNB = confusion_matrix(y_test, X_test_pred)
        print(cmNB)


        st.write(predict_category('receive a free entry'))
        st.write(predict_category('you could win a prize'))
        st.write(predict_category('We will have a meeting'))
        st.write(predict_category('camera for free'))

def predict_category(s):
    pred = clfNB.predict([s])
    return pred

if __name__ == "__main__":
    app()
