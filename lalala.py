import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

#navigasi sidebar
with st.sidebar :
    selected = option_menu('MENU',
    ['Sentiment Analysis',
     'About The Data'],
    default_index=0)#Halaman Sentiment Analysis
if (selected == 'Sentiment Analysis') :
    st.title('Sentiment Analysis of PUBG Mobile Reviews on App Store')
    st.write('This is a sentiment analysis app for PUBG Mobile Reviews on App Store')

    # read csv from a URL
    url = "https://raw.githubusercontent.com/julianfrhn/TA/main/DATASET%20RESTORAN%20(2).csv"
    df = pd.read_csv(url)
    df


    #contoh review
    df = df[df['Rating'] != 3]
    df['sentiment'] = df['Rating'].apply(lambda rating : +1 if rating > 3 else -1)
    positif = df[df['sentiment'] == 1]
    negatif = df[df['sentiment'] == -1]

    #positif
    idx_positif = np.random.randint(0, len(positif))
    contoh_positif = pd.DataFrame(positif[['Review', 'Review']].iloc[idx_positif])
    contoh_positif.columns = ['Example for positive Review']
    st.write(contoh_positif)

    #negatif
    idx_negatif = np.random.randint(0, len(negatif))
    contoh_negatif = pd.DataFrame(negatif[['Review', 'Review']].iloc[idx_negatif])
    contoh_negatif.columns = ['Example for negative Review']
    st.write(contoh_negatif)
    
    kalimat = st.text_area("Input your Review:")

    #model selection
    option_model = st.selectbox(
    'Select the model you want to use :',
    ('Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes'))

    #load model
    vector = pickle.load(open('C:\python_ws\count_vectorizer.sav', 'rb'))
    if option_model == 'Random Forest' :
        model = pickle.load(open(r'C:\python_ws\random_model.sav', 'rb'))
    if option_model == 'Logistic Regression' :
        model = pickle.load(open(r'C:\python_ws\lr_model.sav', 'rb'))
    if option_model == 'SVM' :
        model = pickle.load(open(r'C:\python_ws\svm_model.sav', 'rb'))
    if option_model == 'Naive Bayes' :
        model = pickle.load(open(r'C:\python_ws\nb_model.sav', 'rb'))

    #transform inputan
    kalimat = [kalimat]
    kalimat = vector.transform(kalimat)

    #predict sentiment
    if option_model == 'Naive Bayes' :
        kalimat = kalimat.toarray()
    prediksi = model.predict(kalimat)
    prediksi_proba = model.predict_proba(kalimat)
    if st.button("Analyze Review"):
        if str(kalimat) != '' and prediksi == -1 :
            st.error("The sentiment of your review is negative with probability "+ str(float("{:.2f}".format(prediksi_proba[0][0]*100)))+"%")
        elif str(kalimat) != '' and prediksi == 1:
            st.success("The sentiment of your review is positif with probability "+ str(float("{:.2f}".format(prediksi_proba[0][1]*100)))+"%")
        else :
            st.write("Please input your review")

if (selected == 'About The Data') :
    st.title('About The Data')
    
    #mengambil dataset
    df = pd.read_csv('C:\python_ws\DATASET RESTORAN (2).csv')
    #data review
    st.write('PUBG Mobile app review data from App Store')
    st.write(df)
    
    #review rating bar
    st.write('')
    rating_bar = df.groupby('Rating').agg({'Review':'count'})
    rating_bar.columns = ['Number of Reviews']
    st.bar_chart(rating_bar)

    #wordcloud
    st.write('Wordcloud from the reviews: ')
    wordcloud_review = Image.open('wordcloud11.png')
    st.image(wordcloud_review)
    st.write('Wordcloud of negative reviews: ')
    wordcloud_negatif = Image.open('wordcloud33.png')
    st.image(wordcloud_negatif)
    st.write('Wordcloud of positive reviews: ')
    wordcloud_positif = Image.open('positive.png')
    st.image(wordcloud_positif)
