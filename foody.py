import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

#---------------------------------------------------------------
# 1.Read data
#Source Code
data = pd.read_csv('foody_pos_neg_score.csv')
#------------------------------------------------------------

#GUI
st.title('Data Science Project')
st.write('## Restaurant Review Sentiment Analysis ')
#Upload file
upload_file = st.file_uploader('Choose a file', type = ['csv'])
if upload_file is not None:
    data = pd.read_csv(upload_file)
    data.to_csv('restaurant_review_new.csv', index = False)

def sidebar_bg(side_bg):
    
   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'food.png'
sidebar_bg(side_bg)


# 2. Data pre-processing
source = data['retext']
text_data = np.array(source)
#Label for Class: positive = 1, negative = 0
target = data['sentiment']
target = target.replace('positive',1)
target = target.replace('negative',0)
#Tfidf Vectorizer
tfd = TfidfVectorizer(analyzer='word', max_features=3000)
tfd_model = tfd.fit(text_data)
bag_of_words = tfd.transform(text_data)

X = bag_of_words.toarray()
y = np.array(target)


# 3. Build Model - Machine Learning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) 
#Create model
sgd = SGDClassifier(alpha=7.011524481263914e-05, eta0=0.04748206502813289,
              l1_ratio=0.13849281269604594, loss='log', max_iter=810, n_jobs=1,
              penalty='l1', power_t=0.53255544503547, random_state=3,
              tol=3.110298971318247e-05)
#Fitting training TF_IDF vectors and labels
model = sgd.fit(X_train,y_train)



# 4.Evaluate model
pred = sgd.predict(X_test)


cm = confusion_matrix(y_test,pred, labels=[0,1])
sns.heatmap(cm,square=True, annot=True,fmt='d',cbar=False, cmap='Blues')

cr = classification_report(y_test,pred)

pred_positive = cm[0,1] + cm[1,1]
pred_negative = cm[0,0] + cm[1,0]

actual_positive = cm[1,0] + cm[1,1]
actual_negative = cm[0,0] + cm[0,1]

lst = []
lst.append([pred_positive,pred_negative, actual_positive,actual_negative])
lst = pd.DataFrame(lst, columns = ['pred_positive',
                                   'pred_negative',
                                   'actual_positive',
                                   'actual_negative'])

y_prob = model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

#----------------------------------------------------------------------------------
#GUI
menu = ['Business Objective', 'Build Project','Show Prediction']
choice = st.sidebar.selectbox('Content', menu)
if choice == 'Business Objective':
    st.header('Business Objective')
    st.subheader('Business Understand')
    st.write(""" Building a customer feedback rating system for the restaurant""")
    st.write(""" Goal: The model predicts customer feedback (positive and negative) about restaurant's quality (product and service) that the restaurant understand customers better and improve the quality of food and service, determine the busisness strategy.""")
    st.write(""" Scope: Data have been collect on Food.vn website""")
    st.subheader('Data Understand/ Acquire')
    st.write(""" - Craping data on Foody.vn website.""")
    st.write(""" - Sentiment Analysis in cuisine area.""")
    st.image('rating.jpg', use_column_width='always')
elif choice == 'Build Project':
    st.header('Build Project')
#Show data
    st.subheader("1.Data")
    if st.checkbox("Preview Dataset"):
        st.caption('First 5 lines of dataset')
        st.dataframe(data[['restaurant','sentiment','review_score']].head())
        st.caption('Last 5 lines of dataset')
        st.dataframe(data[['restaurant','sentiment','review_score']].tail())
    if st.checkbox('Info of Dataset'):
        st.caption('Describle of dataset')
        st.write(data.describe())

    st.subheader("2. Data Visualization")
    select = st.selectbox('Visualozation of Review Sentiment',['Amount of Negative & Positive','Amount of Review Score'], key = 1)  
    if select == 'Amount of Negative & Positive':
        gr = data.groupby(by = ['sentiment'])['review_score'].count()
        st.caption('Amount of Negative and Positive')
        st.bar_chart(gr)
    else:
        st.caption('Amount of review score')
        gr1 = data['review_score'].value_counts()
        st.bar_chart(gr1)
    
    st.subheader('2.Build model...')

#Evaluation
    st.subheader("3.Evaluation")
    st.caption('Confusion matrix' )
    st.code(cm)
    fig2 = sns.heatmap(cm,square=True, annot=True,fmt='d',cbar=False, cmap='Blues')
    st.pyplot(fig2.figure)
    st.dataframe(lst)
    st.caption('Classification report')
    st.code(cr)

    #Calulate roc curve
    st.write('ROC curve')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig3, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--")
    ax.plot(fpr,tpr,marker=".")
    st.pyplot(fig3.figure)
    
    st.subheader('4.Summary')
    st.write('This model is good enough for restaurant sentiment analysis - Negative and Positive Classification')

elif choice =='Show Prediction':
    st.header('Show Prediction')
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options = ("Upload", "Input"))
    if type =='Upload':
        uploaded_file_1 = st.file_uploader('Choose a file', type=['txt','csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1,header=None)
            st.dataframe(lines)
            lines = lines[0]
            flag = True
    if type == 'Input':
        with st.form("my_form"):
            email = st.text_area(label = 'Input your content:')
            submitted = st.form_submit_button("Submit")
            if email!="":
                lines = np.array([email])
                flag = True

    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = tfd.transform(lines)        
            y_pred_new = model.predict(x_new)       
            st.code("New predictions (0: Negative, 1: Positive): " + str(y_pred_new))
            #Emoji
            if y_pred_new==0:
                st.markdown('Negative  :angry: ')
            else:
                st.markdown('Positive  :smiley: ')
