import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

# 1. 데이터 불러오기
data = pd.read_csv("spam.csv", encoding="latin1")
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data.dropna(inplace=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2. 텍스트 전처리
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}', '[PHONE]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'[^A-Za-z0-9!$%@#\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['clean_text'] = data['text'].apply(clean_text)
data.drop_duplicates(subset='clean_text', inplace=True)

# 3. 데이터 분리
X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer(
    max_features=5113,
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. SMOTE 적용
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


# 6. 모델 정의 (앙상블)
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr), ('rf', rf)],
    voting='soft'
)

# 7. 학습
ensemble_model.fit(X_train_resampled, y_train_resampled)

# 8. 평가
y_pred = ensemble_model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n[모델 성능 평가]")
print(f"정확도 (Accuracy): {acc:.4f}")
print(f"정밀도 (Precision): {prec:.4f}")
print(f"재현율 (Recall): {rec:.4f}")
print(f"F1 점수 (F1-score): {f1:.4f}")

# 9. 모델 및 벡터라이저 저장
joblib.dump(ensemble_model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n 모델 학습 및 저장 완료! (spam_model.pkl, vectorizer.pkl)")
