import pandas as pd
from sqlalchemy import create_engine
import joblib
import matplotlib.pyplot as plt
from nltk.tokenize  import RegexpTokenizer
import re
from nltk.corpus import stopwords
import nltk as nk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer



class Prediction():
    def __init__(self, userEngine='', passwordEngine='', hostEngine='',dataBaseEngine='', modelName = ''):
        self.userEngine = 'root'
        self.passwordEngine = ''
        self.hostEngine = '127.0.0.1'
        self.portEngine = 3306
        self.modelName = 'outputModel/modelNaiveBayesJl.pkl'
        self.dataBaseEngine = 'db_sma2'

    
    def engine(self):
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(self.userEngine, self.passwordEngine, self.hostEngine, self.portEngine, self.dataBaseEngine)
        return url


    def readData(self,nameTable):
        try:
            engine = self.engine()
            df = pd.read_sql(f"select * from {nameTable} LIMIT 4000", engine)
            return df
        except Exception as ex:
            return print("Error: \n", ex)

    def predictedOwn(self,ds):
        vect = TfidfVectorizer(max_features=5600)
        xNewDataset = vect.fit_transform(ds)
        loaded_model = joblib.load(self.modelName)
        p = loaded_model.predict(xNewDataset)

        dos = []

        for prediction in p:
            if prediction == "NETRAL":
                do = 0
            elif prediction == "POSITIF":
                do = 1
            else:
                do = 2
            dos.append(do) 

        return dos

class PreProcessing(Prediction):

    def __init__(self):
        super(PreProcessing, self).__init__()

    def caseFolding(self, kalimat):
        kalimat = kalimat.strip()
        kalimat = kalimat.lower()
        kalimat = re.sub(r'[|?|$|.|!_:")(-+,)]','', kalimat)
        return kalimat
    
    def tokenizing(self):
        regexp = RegexpTokenizer(r'\w+|$[0-9]+|\S')
        return regexp
    
    def stopWord(self, text):
        stopword = stopwords.words('indonesian')
        txt_stopword = pd.read_csv('text/stopword.txt', names=['stopword'], header=None)
        stopword.extend(['wkwk','hahahaha','haha','yang','yoi','yoyoy', 'mm', 'mk', 'bandung', 'subang','wartakinico','galamedianews'])
        stopword.extend(txt_stopword["stopword"][0].split('\n'))
        stopword = set(stopword)

        text = [word for word in text if word not in stopword]
        return text

    def stemmingWithjoin(self, konten):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        do = []
        for i in konten:
            dt = stemmer.stem(i)
            do.append(dt)
        
        d_clean = []
        d_clean = " ".join(do)
        return d_clean
    
    def stemming(self):
        pass



