from collections import Counter
import pandas as pd
import os 
import pandas as pd

import re
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.decomposition import NMF, LatentDirichletAllocation

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

import textblob as tb
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
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
from nltk.corpus import wordnet
from nltk import pos_tag

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



def compact(lst):  #remove empty strings from a list
    return list(filter(None, lst))

# compact([0, 1, False, 2, '', 3, 'a', 's', 34]) #Sample
def convert(lst):
    return ([i for item in lst for i in item.split()])
    with open('Data_Stop_Words_Storage.csv', 'r') as f:
        stop_words = f.read().strip().split(',')  # we want to split the data on comma after this operation it will be stored as list
#     stop_words=str(stop_words) #cant perform splitting on list so converting into str and then back again to list
#     stop_words = stop_words.split() 
#     stop_words=list(stop_words)
    stop_words=convert(stop_words)
    stop_words= compact(stop_words)
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
tf_idf.iloc[:,500:600].head()



lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(Work_Data['Dependant Variable'])
y=pd.DataFrame(y,columns=['Category'])



X_train, X_test, y_train, y_test = train_test_split(tf_idf, y, test_size=0.15, random_state=43,stratify=y)

#data['sentiment_score'] this is our target variable (1 and 0 already defined) as (Y) and tf_idf as our independant features
print (f'Train set shape\t:{X_train.shape}\nTest set shape\t:{X_test.shape}')


ytrain_data = pd.DataFrame(y_train)
train_data = pd.concat([X_train, ytrain_data],axis=1)
train_data[:5] 

model=MultinomialNB()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

sample=['miecsha williams call everyday ebb credit miecsha hope well still result week call several time day ridiculous auto generate system message user return b application number allow minute look sure thank available another minute question sure thank enrol temporary federal emergency broadband benefit program upcoming bill soon see credit apply cost current internet service lease modem note credit may appear second bill enrollment case receive two credit account first second month enrol program ’ continue apply credit bill month offset internet service charge lease modem ’ general information although government yet announce program end date ’ keep updated regularly month ahead provide least day notice program concludes point service continue regular monthly charge resume unless choose change cancel service ’ apply still pending ’ approve yes check confirm apply ebb credit pending status change approve assure also ’ bill amount cuz ’ state something different end check regular monthly bill agent say bill say yes right go forward adjust ebb discount ’ state yes adjustment yet make notify adjustment tell prorated end amount ebb discount yet apply account applied account notify assured speaking ebb bill amount amount actual amount right allow minute check current balance account exactly ’ show end tho yes take time update end update real time thank ’ sll customer close windowconnecting miecsha williams hey miecsha today bill say ’ get concern thank much bring concern u happy help allow minute quickly check sure thank time available another minute question sure thank much wait meicsha check account see bill amount able see incorrect check correct bill amount pro rat credit rest assure quickly escalate dedicate team get update quickly possible thank much understanding anything else make day well also check ebb credit sure quickly check thank much wait miecsha check ebb application account process application get approved start get credit account approve receive email last week yes absolutely right approve however check process allow minute quickly check estimated date start get credit thank much wait see within business day application successfully process start get credit upcoming bill also apply ebb month receive double credit upcoming bill one month upcoming month business day cuz ’ approve within business day completely agree point application approve ebb credit add account successfully process within business day ’ see cuz tell ’ see ’ call everyday rest assure meicsha check application successfully process within business day thank much understand anything else make day well okay customer close']
encode=vectorizer.transform(sample)
# prediction_value=model.predict(encode)
# print ('Prediction Output:' , lbl_enc.inverse_transform(prediction_value))



import pickle
test_mod=pickle.load(open('SaveMod.pkl','rb'))
prediction=test_mod.predict(encode)
print ('Prediction Output:' , lbl_enc.inverse_transform(prediction))