import pickle

import nltk
from flask import Flask, render_template, request
from nltk import WordNetLemmatizer
from nltk.corpus.reader import wordnet

#Function to get Part of Speech of the text
def get_wordnet_pos_tag(word):
  tag = nltk.pos_tag([word])[0][1][0]
  tag_dict = {
      "J":wordnet.ADJ,
      "R":wordnet.ADV,
      "N":wordnet.NOUN,
      "V":wordnet.VERB
  }
  return tag_dict.get(tag, wordnet.NOUN)

#Function to tokenize lemma
def tokenize_lemma(text):  # User defined or custom tokenizer using lemma -- To add POS part
  tokens = nltk.word_tokenize(text)
  lemma = WordNetLemmatizer()

  clean_tokens = []
  for token in tokens:
    # print(lemma.lemmatize(token))
    # lemm.lemmatize(word,get_wordnet_pos_tag(word))
    clean_tokens.append(lemma.lemmatize(token,get_wordnet_pos_tag(token)))
  return clean_tokens


app = Flask(__name__)

#Depickling the model
clf = pickle.load(open('model.pkl', 'rb'))

#Depickling the Count Vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

#Depickling the TFIDF vectorizer
with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

#Depickling the Label Encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route("/") #decorators
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    #label encode and normalize the inputs
    features = request.form.values()

    output = clf.predict(tfidf.transform(vectorizer.transform(features)))
    print(f'Classfier output = {output}')
    tweet_sentiment = label_encoder.inverse_transform(output)
    print(f'decoded output = {tweet_sentiment}')
    if tweet_sentiment == 'positive':
        pred =  'positive'
    elif tweet_sentiment == 'negative':
        pred = 'negative'
    else:
        pred =  'neutral'

    return  render_template('index.html', pred = pred)


if __name__ == "__main__":
    app.run(debug = True)