# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from nlp_engine.preprocessing import clean_text_

# class IntentClassifier:
#     def __init__(self, model_path="models/best_model.pkl", vectorizer_path="models/vectorizer.pkl"):
#         self.model = joblib.load(model_path)
#         self.vectorizer = joblib.load(vectorizer_path)
    

#     def predict_intent(self, user_input):
#         cleaned = clean_text_(user_input)
#         X = self.vectorizer.transform([cleaned])
        
       

#         return self.model.predict(X)[0]
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nlp_engine.preprocessing import clean_text_

class IntentClassifier:
    def __init__(
        self,
        model_path="models/best_model.pkl",
        vectorizer_path="models/vectorizer.pkl",
        label_encoder_path="models/label_encoder.pkl"
    ):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict_intent(self, user_input):
        cleaned = clean_text_(user_input)
        X = self.vectorizer.transform([cleaned])
        label_index = self.model.predict(X)[0]  # This is a numeric label (e.g., 2)
        intent_name = self.label_encoder.inverse_transform([label_index])[0]
        return intent_name
