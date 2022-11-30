import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("Data Mining")
st.write("Nama  : Dhita Aprilia Dhamayanti")
st.write("NIM   : 200411100102")
upload_data, desc, preporcessing, modeling, implementation = st.tabs(["Upload Data","Description", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah data Klasifikasi Kardiovaskuler dari https://www.kaggle.com/code/ekramasif/cardiovasculardiseasepredictionusingml")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with desc:
    st.write("Dataset ini adalah dataset menengenai penyakit cardiovascular. Kardiovaskular atau Cardiovascular Disease (CVD) merupakan penyakit yang berkaitan dengan jantung dan pembuluh darah.")
    st.write("Dataset ini diambil dari https://www.kaggle.com/code/ekramasif/cardiovasculardiseasepredictionusingml")
    st.write("""# Features""")
    st.write("Age, Height, Weight, Gender, Ap_hi (Systolic blood pressure), Ap_lo (Diastolic blood pressure), Cholesterol, Gluc (Glucose), Smoke, Alco (Alcohol), Active (Physical activity).")
    st.write("Aplikasi ini berfungsi untuk menentukan penyakit Cardiovascular.")
    st.write("Source Code dapat diakses melalui ")

with preporcessing:
    st.write("""# Preprocessing""")
    df[["id", "age", "gender", "height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke"]].agg(['min','max'])

    df.cardio.value_counts()

    X = df.iloc[:,1:-1]
    y = df.iloc[:, -1]
    X
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.cardio).columns.values.tolist()

    "### Label"
    labels

    st.markdown("# Normalize")

    "### Normalize data"

    dataubah=df.drop(columns=["id", "age", "gender", "height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke"])
    dataubah

    "### Normalize data gender"
    data_id=df[['id']]
    id = pd.get_dummies(data_id)
    id

    # "### Normalize data Hypertension"
    # data_hypertension=df[['hypertension']]
    # hypertension = pd.get_dummies(data_hypertension)
    # hypertension

    "### Normalize data age"
    data_age=df[['age']]
    age = pd.get_dummies(data_age)
    age

    "### Normalize data gender"
    data_gender=df[['gender']]
    gender = pd.get_dummies(data_gender)
    gender

    "### Normalize data height"
    data_height=df[['height']]
    height = pd.get_dummies(data_height)
    height

    "### Normalize data weight"
    data_weight=df[['weight']]
    weight = pd.get_dummies(data_weight)
    weight

    "### Normalize data ap_hi"
    data_ap_hi=df[['ap_hi']]
    ap_hi = pd.get_dummies(data_ap_hi)
    ap_hi

    "### Normalize data ap_lo"
    data_ap_lo=df[['ap_lo']]
    ap_lo = pd.get_dummies(data_ap_lo)
    ap_lo

    "### Normalize data cholesterol"
    data_cholesterol=df[['cholesterol']]
    cholesterol = pd.get_dummies(data_cholesterol)
    cholesterol

    "### Normalize data gluc"
    data_gluc=df[['gluc']]
    gluc = pd.get_dummies(data_gluc)
    gluc

    "### Normalize data smoke"
    data_smoke=df[['smoke']]
    smoke = pd.get_dummies(data_smoke)
    smoke

    dataOlah = pd.concat([id, age, gender, height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke], axis=1)
    dataHasil = pd.concat([df,dataOlah], axis = 1)

    X = dataHasil.drop(columns=["id", "age", "gender", "height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke"])
    y = dataHasil.cardio
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(dataHasil.cardio).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape



with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

# with modeling:

#     st.markdown("# Model")
#     # membagi data menjadi data testing(20%) dan training(80%)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#     # X_train.shape, X_test.shape, y_train.shape, y_test.shape

#     nb = st.checkbox("Metode Naive Bayes")
#     knn = st.checkbox("Metode KNN")
#     dt = st.checkbox("Metode Decision Tree")
#     sb = st.button("submit")

#     #Naive Bayes
#     # Feature Scaling to bring the variable in a single scale
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     GaussianNB(priors=None)
#     # Fitting Naive Bayes Classification to the Training set with linear kernel
#     nvklasifikasi = GaussianNB()
#     nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

#     # Predicting the Test set results
#     y_pred = nvklasifikasi.predict(X_test)
        
#     y_compare = np.vstack((y_test,y_pred)).T
#     nvklasifikasi.predict_proba(X_test)

#     akurasi = round(100 * accuracy_score(y_test, y_pred))

#     #Decision tree
#     dt = DecisionTreeClassifier()
#     dt.fit(X_train, y_train)

#     # prediction
#     dt.score(X_test, y_test)
#     y_pred = dt.predict(X_test)
#     #Accuracy
#     akur = round(100 * accuracy_score(y_test,y_pred))

#     K=10
#     knn=KNeighborsClassifier(n_neighbors=K)
#     knn.fit(X_train,y_train)
#     y_pred=knn.predict(X_test)

#     skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    

#     if nb:
#         if sb:

#             """## Naive Bayes"""
            
#             st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))

#     if knn:
#         if sb:
#             """## KNN"""

#             st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    
#     if dt:
#         if sb:
#             """## Decision Tree"""
#             st.write('Model Decission Tree Accuracy Score: {0:0.2f}'.format(akur))

with implementation:
    st.write("# Implementation")
   
    # Sex = st.radio(
    # "Masukkan Jenis Kelamin Anda",
    # ('Laki-laki','Perempuan'))
    # if Sex == "Laki-laki":
    #     Sex_Female = 0
    #     Sex_Male = 1
    # elif Sex == "Perempuan" :
    #     Sex_Female = 1
    #     Sex_Male = 0

    # BP = st.radio(
    # "Masukkan Tekanan Darah Anda",
    # ('Tinggi','Normal','Rendah'))
    # if BP == "Tinggi":
    #     BP_High = 1
    #     BP_LOW = 0
    #     BP_NORMAL = 0
    # elif BP == "Normal" :
    #     BP_High = 0
    #     BP_LOW = 0
    #     BP_NORMAL = 1
    # elif BP == "Rendah" :
    #     BP_High = 0
    #     BP_LOW = 1
    #     BP_NORMAL = 0

    # Cholesterol = st.radio(
    # "Masukkan Kadar Kolestrol Anda",
    # ('Tinggi','Normal'))
    # if Cholesterol == "Tinggi" :
    #     Cholestrol_High = 1
    #     Cholestrol_Normal = 0 
    # elif Cholesterol == "Normal":
    #     Cholestrol_High = 0
    #     Cholestrol_Normal = 1
        
    # Na_to_K = st.number_input('Masukkan Rasio Natrium Ke Kalium dalam Darah')



   

