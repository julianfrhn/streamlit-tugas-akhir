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
     'About The Data',
     'About The Application'],
    default_index=0)#Halaman Sentiment Analysis
if (selected == 'Sentiment Analysis') :
    st.title('SENTIMEN ANALISIS REVIEW RESTORAN BAKMIE DI BANDUNG MENGGUNAKAN METODE MACHINE LEARNING')
    st.write('This is a sentiment analysis app for Bakmie Restaurant Review in Bandung')
    url = "https://raw.githubusercontent.com/julianfrhn/streamlit-tugas-akhir/main/DATASET%20RESTORAN%20(2).csv"
    df = pd.read_csv(url)

    #contoh review
    df = df[df['Rating'] != 3]
    df['sentiment'] = df['Rating'].apply(lambda rating : +1 if rating > 3 else -1)
    positif = df[df['sentiment'] == 1]
    negatif = df[df['sentiment'] == -1]

    #positif
    idx_positif = np.random.randint(0, len(positif))
    contoh_positif = pd.DataFrame(positif[['Nama Reviewer', 'Review']].iloc[idx_positif])
    contoh_positif.columns = ['Example for positive Review']
    st.write(contoh_positif)

    #negatif
    idx_negatif = np.random.randint(0, len(negatif))
    contoh_negatif = pd.DataFrame(negatif[['Nama Reviewer', 'Review']].iloc[idx_negatif])
    contoh_negatif.columns = ['Example for negative Review']
    st.write(contoh_negatif)
    
    kalimat = st.text_area("Input your Review:")

    #model selection
    option_model = st.radio(
    'Select the model you want to use :',
    ('Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes'))

    #load model
    vector = pickle.load(open('count_vectorizer.sav','rb'))
    if option_model == 'Random Forest' :
        model = pickle.load(open(r'random_model.sav', 'rb'))
    if option_model == 'Logistic Regression' :
        model = pickle.load(open(r'lr_model.sav', 'rb'))
    if option_model == 'SVM' :
        model = pickle.load(open(r'svm_model2.sav', 'rb'))
    if option_model == 'Naive Bayes' :
        model = pickle.load(open(r'nb_model.sav', 'rb'))

    #transform inputan
    kalimat = [kalimat]
    kalimat = vector.transform(kalimat)

    #predict sentiment
    if option_model == 'Naive Bayes' :
        kalimat = kalimat.toarray()
    prediction = model.predict(kalimat)
    prediction_proba = model.predict_proba(kalimat)
    if st.button("Analyze Review"):
        if str(kalimat) != '' and prediction == -1 :
            st.error("The sentiment of your review is negative with probability "+ str(float("{:.2f}".format(prediction_proba[0][0]*100)))+"%")
        elif str(kalimat) != '' and prediction == 1:
            st.success("The sentiment of your review is positif with probability "+ str(float("{:.2f}".format(prediction_proba[0][1]*100)))+"%")
        else :
            st.write("Please input your review")

if (selected == 'About The Data') :
    st.title('About The Data')
    
    #mengambil dataset
    url = "https://raw.githubusercontent.com/julianfrhn/streamlit-tugas-akhir/main/DATASET%20RESTORAN%20(2).csv"
    df = pd.read_csv(url)
    

    #data review
    st.write('Bakmie Restaurant Review data in Bandung')
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

if (selected == 'About The Application') :
    st.title('About The Application')

    st.write('Tujuan Dari Aplikasi Ini')
    st.write('Di era web seperti sekarang, sejumlah informasi kini mengalir melalui jaringan. Karena berbagai konten web meliputi opini subjektif serta informasi yang objektif, sekarang umum bagi orang-orang untuk mengumpulkan informasi tentang produk dan jasa yang mereka ingin beli. Namun karena cukup banyak informasi yang ada dalam bentuk teks tanpa ada skala numerik, sulit untuk mengklasifikasikan evaluasi informasi secara efisien tanpa membaca teks secara lengkap. Maka Tujuan dari aplikasi ini adalah untuk mempermudah pengguna dalam mencari informasi mengenai Restoran Bakmie di Bandung.')