import tensorflow as tf 
import streamlit as st 
import joblib 
import sklearn
import nltk 
import re
import string
from nltk.stem import WordNetLemmatizer
import base64



css = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://th.bing.com/th/id/OIP.m3a3mLSWSsvTrMAyRTSR5QHaE8?w=268&h=180&c=7&r=0&o=5&dpr=1.8&pid=1.7");
background-size: 120%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stExpander"] {{
background: rgba(0,0,0,0.5);
border: 2px solid #000071;
border-radius: 10px;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)






# check if wordnet is installed
try:
    nltk.find("corpora/popular.zip")
except LookupError:
    nltk.download("popular")

# read sw_new txt file
with open("sw_new.txt", "r") as f:
    sw_new = f.read()
sw_new = sw_new.split("\n")

# define text cleaner function
def text_cleaner(text, sw = sw_new):
  import nltk
  import re
  # import emoji
  import string
  from nltk.stem import WordNetLemmatizer

  # mobile_regex = "(\+*)((0[ -]*)*|((91 )*))((\d{12})+|(\d{10})+)|\d{5}([- ]*)\d{6}"
  url_regex = "((http|https|www)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
  space_regex = "\s\s+"
  # remove url
  text = re.sub(url_regex, "", text)
  # remove mobile
  # text = re.sub(mobile_regex, "", text)
  # lower casing
  text = text.lower()
  # remove emoji & punctuation & numbers
  text = "".join([i for i in text if (ord(i) in range(97,123)) | (i == " ")])
  # remove multiple spaces
  text = re.sub(space_regex, " ", text)

  # stopword removal
  text = [i for i in text.split() if i not in sw]
  # lemmatizing
  lemma = WordNetLemmatizer()
  text = " ".join([lemma.lemmatize(i) for i in text])

  return text

# load model
@st.cache_resource
def catch_model(model1_add):
    model1 = tf.saved_model.load(model1_add)
    return model1

nlp_model = catch_model("Flipkart_review_use_model")


def prediction(text):
  text = text_cleaner(text)
  y_pred_nlp = nlp_model([text]).numpy()
  return y_pred_nlp
#   print(("Negative Review" if y_pred[0] == 1 else "Positive Review"))


# UI
# project title,display Banner, display taxt box
st.markdown("<h1 style='text-align: center; color: black;'>Flipkart Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.image('https://global-uploads.webflow.com/5ab25784c7fcbff004fa8dca/5ec3b0b58737c8c8d53e5fe1_ST0-1-360p-cc78b25d-faec-47ee-a379-51dd9aea5e37.gif')
review = st.text_area(
   "Enter pr paste a food review to analyze"
)

# presd Button

pred = st.button("Get_Prediction")

if pred:
    if review:
        y_pred_nlp = prediction(review)
        nlp_result = ("Negative Review" if y_pred_nlp[0,0] >= 0.5 else "Positive Review")
        if nlp_result == "Positive Review":
            st.write(
            f"NLP Model Output: {nlp_result} with {round((1 - y_pred_nlp[0,0])*100, 2)}% Probability"
            )
        else:
            st.write(
                f"NLP Model Output: {nlp_result} with {round(y_pred_nlp[0,0]*100, 2)}% Probability"
            )

    else:
        st.write("Please enter review")   
else:
    pass       