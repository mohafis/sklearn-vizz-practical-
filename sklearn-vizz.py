import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

data = pd.read_csv("PRU14.csv")
    
st.title('MALAYSIAN GENERAL ELECTION 14 ANALYSIS')

st.sidebar.write("""
This is a web app practical to analyse the General Election 14 results by utilizing python libraries such as Streamlit, Sklearn etc.
""")

st.sidebar.write ("This analysis is produced by:")

st.sidebar.write("<a href='https://www.linkedin.com/in/mohd-hafizzudin-ismail-92b15274/'>Mohd Hafizzudin Ismail aka mohafis</a>", unsafe_allow_html=True)


choice = st.sidebar.radio(
    "Choose between two option",   
    ('Analysis', 'Machine Learning'),
    index = 0
    )

st.write(f"## You Have Selected <font color='Gold'>{choice}</font> Option", unsafe_allow_html=True)
    
if choice == 'Analysis':
    st.write("## 1: Top 5 most voted CANDIDATE")
        
    max_voted_candidate = data.groupby('NAMA CALON')['BILANGAN UNDI'].sum()
    max_voted_candidate = pd.DataFrame(max_voted_candidate)
    max_voted_candidate = max_voted_candidate.sort_values(by= 'BILANGAN UNDI',ascending=False)
    top5_max_voted_candidate = max_voted_candidate.head(5)
    top5_max_voted_candidate = top5_max_voted_candidate.reset_index()
        
    #top5_max_voted_candidate.keys()
    top5_max_voted_candidate
        
    plt.figure(figsize = (12,5))
    plt.bar(top5_max_voted_candidate['NAMA CALON'],top5_max_voted_candidate['BILANGAN UNDI'], color=[ '#00A19C', '#20419A', '#763F98', '#FDB924', '#BFD730'])

    st.write("## 2: Top 5 most voted PARTY")
        
    max_voted_party = data.groupby('PARTI')['BILANGAN UNDI'].sum()
    max_voted_party = pd.DataFrame(max_voted_party)
    top5_max_voted_party = max_voted_party.sort_values(by= 'BILANGAN UNDI',ascending=False)
    top5_max_voted_party = top5_max_voted_party.head(5)
    top5_max_voted_party = top5_max_voted_party.reset_index()
        
    top5_max_voted_party

    plt.figure(figsize = (12,5))
    plt.bar(top5_max_voted_party['PARTI'],top5_max_voted_party['BILANGAN UNDI'], color=[ '#00A19C', '#20419A', '#763F98', '#FDB924', '#BFD730'])


else:
    st.write("")
