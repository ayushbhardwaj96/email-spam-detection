from taipy.gui import Gui, notify
import os
import pickle
import string
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
if os.path.exists(NLTK_DIR) and NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

for resource, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ''.join(ch if ch.isalpha() or ch.isspace() else ' ' for ch in text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [lemma.lemmatize(word) for word in tokens]
    return " ".join(tokens)
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

intro = "Enter the message:"
dummy = " "
message = " "
h_text = "Click to classify the spam or ham mail/sms"

page = """
<|text-center|
#
# Email/SMS **Spam**{:.color-primary} Detector
|>

<|{intro}|>

#

<|{dummy}|input|multiline|class_name=fullwidth|label=Hi there! Rohan Sharma this side!|>

#
<|text-center|

<|Predict|button|on_action=on_button_action|hover_text = Click to classify the spam or ham mail/sms|center|>

#

<|{message}|input|not active|label= Know the results here!|>


######Made with **LOVE**{:.color-primary} by Rohan Sharma
|>

"""
def on_button_action(state):
    notify(state, 'info', f'Results may be inaccurate due to limited dataset. Thanks for using this Application :p')
    on_change(state)

def on_change(state):
    if not state.dummy.strip():
        state.message = "Please enter a message"
        return
    # 1. preprocess
    transformed_sms = transform_text(state.dummy)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        state.message = "Spam"       
    else:
       state.message ="Not-Spam"

if __name__ == "__main__":
    app = Gui(page).run(watermark="""Taipy Inside :)""", use_reloader=False)