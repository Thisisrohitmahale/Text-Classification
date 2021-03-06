from flask import Flask,render_template, request
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from nltk.corpus import wordnet
import textblob as tb
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import re
import pickle


df= pd.read_excel('Main_Data_from_Origin & Repeat_Categorized.xlsx',dtype=str) 
df.drop('Unnamed: 0',axis=1,inplace=True)
df_due_payment_top10=df[df['Dependant Variable']=='Due_Payment'].head(10)
df_other_categories_top20=df[df['Dependant Variable']!='Due_Payment'].head(20)
df= pd.concat([df_due_payment_top10,df_other_categories_top20])
df.reset_index(drop=True,inplace=True)


df['Concatenated'] = df['Original Transcript']+df['Repeat Transcript'] 

df=df[['Orig UCID (Voice|Chat)','Repeat UCID (Voice|Chat)','Original Transcript','Repeat Transcript','Concatenated','Dependant Variable']]

send=df.to_excel('Main_file_Origin_Repeat_Concatenated.xlsx')
main_data= pd.read_excel('Main_file_Origin_Repeat_Concatenated.xlsx')
main_data.reset_index(drop=True, inplace=True)
main_data=main_data.iloc[:,1:]
Concatenated_Main_Data=main_data.iloc[:,4::]
StopWords=nltk.corpus.stopwords.words('english')
StopWords[:5]



# def compact(lst):  #remove empty strings from a list
#     return list(filter(None,lst))

# compact([0, 1, False, 2, '', 3, 'a', 's', 34]) #Sample
def convert(lst):
    return ([i for item in lst for i in item.split()])

with open('Data_Stop_Words_Storage.csv', 'r') as f:
    stop_words = f.read().strip().split(',')  # we want to split the data on comma after this operation it will be stored as list
    stop_words=str(stop_words) #cant perform splitting on list so converting into str and then back again to list
    stop_words = stop_words.split() 
    stop_words=list(stop_words)
    stop_words=convert(stop_words)
    # stop_words= compact(stop_words)
    stop_words[-5:]
    
    StopWords.extend(stop_words)




def clean_text(text):
    text=str(text)
    
    text = text.lower()
    text =text.strip() #removing white space from both the ends
    text=re.compile(r'<.*?>').sub('',text) #replacing angular bracket with blank
    text=re.compile(r'\S*@\S*\s?').sub('',text)# removing email ids
    text = re.sub(r'\d+',' ', text) #removing digits
    text= re.compile(r'[%s]'% re.escape(string.punctuation)).sub(' ',text) #removing special charactors
    text =re.sub(r'\s+', ' ',text) # removing white space
    text=text.strip()
    return text

def remove_stopwords(text):
    
    text=str(text)
    filtered_sentence = []
    words =wordpunct_tokenize(text)
    for w in words:  # if the word is present in the words list (all the text in a column) and if that same word is not 
        if w not in StopWords: #present in StopWords then add that word to filtered sentence list
            filtered_sentence.append(w)
    text = ' '.join(filtered_sentence)
    return text

def get_wordnet_pos(tag):
    if tag.startswith ('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(text):
    
    text =str (text)

    wl = WordNetLemmatizer()

    lemmatized_sentence=[]

    words = wordpunct_tokenize(text) 

    word_pos_tags = nltk.pos_tag(words)

    for idx, tag in enumerate(word_pos_tags):

        lemmatized_sentence.append(wl.lemmatize(tag[0],get_wordnet_pos(tag[1])))
        
    lemmatized_text = ' '.join(lemmatized_sentence)
    return lemmatized_text


Concatenated_Main_Data['Clean_Text']=Concatenated_Main_Data['Concatenated'].apply(clean_text)
Concatenated_Main_Data['Clean_Text']=Concatenated_Main_Data['Clean_Text'].apply(remove_stopwords)
Concatenated_Main_Data['Clean_Text']=Concatenated_Main_Data['Clean_Text'].apply(lemmatize)
Concatenated_Main_Data=Concatenated_Main_Data[['Concatenated','Clean_Text','Dependant Variable']]

Concatenated_Main_Data['length']=Concatenated_Main_Data.Concatenated.str.split().apply(len)
# Concatenated_Main_Data['length']=(Concatenated_Main_Data['Concatenated']).apply(len)
Concatenated_Main_Data.head()



Work_Data=Concatenated_Main_Data.iloc[:,[1,2]]
vectorizer = TfidfVectorizer(max_features=None,ngram_range=(1, 2))

features = vectorizer.fit_transform(Work_Data['Clean_Text'])
tf_idf = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names()) #creating the new dataframe with word features
# tf_idf.iloc[:,500:600].head()


lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(Work_Data['Dependant Variable'])
y=pd.DataFrame(y,columns=['Category'])

y= np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(tf_idf, y, test_size=0.15, random_state=43,stratify=y)

#data['sentiment_score'] this is our target variable (1 and 0 already defined) as (Y) and tf_idf as our independant features
print (f'Train set shape\t:{X_train.shape}\nTest set shape\t:{X_test.shape}')


ytrain_data = pd.DataFrame(y_train)
train_data = pd.concat([X_train, ytrain_data],axis=1)
train_data[:5] 
model=MultinomialNB()
model.fit(X_train, y_train)
print(model.score(X_test,y_test))

import pickle
pickle.dump(model,open('SaveMod.pkl','wb'))
# pick_model= pickle.load(open('SaveMod.pkl','rb'))


app= Flask(__name__)


# @app.route('/')
# def hello_word():
#     return 'Hello Everyone'

@app.route('/')
def new_route():
    return render_template('Home.html')

@app.route('/predict',methods=['Get','Post'])

def predict():

    pick_model= pickle.load(open('SaveMod.pkl','rb'))

    input_str = request.form.get('transcript')
    # print (type(input_str))
    input_str=clean_text(input_str)
    input_str=remove_stopwords(input_str)
    input_str=lemmatize(input_str)
    input_str= ([input_str])
    print (input_str)
    encoded_str= vectorizer.transform(input_str)
    # return (str(input_str))
    output= pick_model.predict(encoded_str)
    out = lbl_enc.inverse_transform(output)
    print (out)
    return render_template('output.html',out=out)



# Run the app
if __name__=='__main__':
    app.run(debug=True)