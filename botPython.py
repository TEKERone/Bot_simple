import nltk
import numpy as np
import random
import string # to process standard python strings

f=open('chatbot.txt','r',errors = 'ignore')

raw=f.read()

raw=raw.lower()# se convierte a minúsculas

nltk.download('punkt') # solo uso por primera vez
nltk.download('wordnet') # first-time use only

sent_tokens = nltk.sent_tokenize(raw)# cconvierte a la lista de oraciones
word_tokens = nltk.word_tokenize(raw)# se convierte a la lista de palabras

sent_tokens[:2]
['un chatbot (también conocido como talkbot, chatterbot, bot, im bot, agente interactivo o entidad conversacional artificial) es un programa de computadora o inteligencia artificial que realiza una conversación a través de métodos auditivos o textuales.',
'tales programas a menudo están diseñados para simular de manera convincente cómo se comportaría un ser humano como un compañero de conversación, y así pasará la prueba de Turing. ']


word_tokens[:2]
['un', 'chatbot', '(', 'also', 'known', 'a', 'chatbot', '(', 'también', 'conocido']

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet es un diccionario de inglés de orientación semántica incluido en NLTK.

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hola","hi", "saludos", "sup", "que pasa","hey")

GREETING_RESPONSES = ["hi", "hey", "* asiente con la cabeza *", "Hola", "¡Me alegro! Usted está hablando conmigo"]

def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        robo_response=robo_response+"Lo siento! No te entiendo!!"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("ChatBot: Hola soy un ChatBot. Voy a responder a sus consultas sobre Chatbots. Si quieres salir, escribe Bye!")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='gracias'):
            flag=False
            print("ChatBot: De nada, Nos vemos . . .")
        else:
            if(greeting(user_response)!=None):
                print("ChatBot: "+greeting(user_response))
            else:
                print("ChatBot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ChatBot: Adios! Cuidate . . .")
