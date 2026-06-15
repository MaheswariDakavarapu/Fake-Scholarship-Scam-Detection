# ---------------- IMPORTS ----------------
import pandas as pd
import re
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ---------------- SCHOLARSHIP KEYWORDS ----------------
SCHOLARSHIP_KEYWORDS = [
    "scholarship", "stipend", "aicte", "ugc", "nsp",
    "government", "govt", "fee", "education", "grant",
    "post matric", "pre matric","Vidya", "Deevena"
]

def is_scholarship_message(message):
    message = clean_text(message)
    return any(word in message for word in SCHOLARSHIP_KEYWORDS)

# ---------------- TRUSTED DOMAINS ----------------
TRUSTED_DOMAINS = [
    "gov.in", "ac.in", "edu", "org", "nic.in", "co.in"
]

def has_trusted_domain(message):

    urls = re.findall(r'(https?://\S+|www\.\S+)', message.lower())

    for url in urls:
        for domain in TRUSTED_DOMAINS:
            if domain in url:
                return True

    return False

# ---------------- FAKE KEYWORDS ----------------
FAKE_KEYWORDS = [
    "pay fee", "processing fee", "registration fee",
    "instant approval", "guaranteed scholarship",
    "limited seats", "apply immediately",
    "send money", "whatsapp payment"
]

# ---------------- GENUINE KEYWORDS ----------------
GENUINE_KEYWORDS = [
    "education", "foundation",
    "government", "portal", "scheme", "grant","vidya","Deevena","vasathi"

]

# ---------------- RULE BASED DETECTION ----------------
def rule_based_detection(message):

    msg = message.lower()

    # Fake keyword detection
    for word in FAKE_KEYWORDS:
        if word in msg:
            return "FAKE", 0.90

    # Trusted domain detection
    if has_trusted_domain(message):
        return "GENUINE", 0.92

    # Genuine keyword detection
    for word in GENUINE_KEYWORDS:
        if word in msg:
            return "GENUINE", 0.80

    return None, None

# ---------------- CREATE DATA FOLDER ----------------
if not os.path.exists("data"):
    os.makedirs("data")

# ---------------- LOAD ML DATASET ----------------
data = pd.read_csv("data/scholarship_admission_dataset.csv")

data["message"] = data["message"].apply(clean_text)

X = data["message"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

# ---------------- FEEDBACK FILE ----------------
feedback_file = os.path.join("data", "feedback.csv")

if not os.path.exists(feedback_file):
    pd.DataFrame(columns=[
        "timestamp",
        "message",
        "predicted_label",
        "confidence",
        "link",
        "satisfaction",
        "rating"
    ]).to_csv(feedback_file, index=False)

# ---------------- LOAD SCHOLARSHIPS ----------------
scholarships_file = "data/scholarships_india.csv"

if os.path.exists(scholarships_file):
    scholarships = pd.read_csv(scholarships_file)
else:
    scholarships = pd.DataFrame(columns=["name","link"])

# ---------------- FIND SCHOLARSHIP LINK ----------------
def find_scholarship_link(message):

    message_words = set(clean_text(message).split())

    for _, row in scholarships.iterrows():

        name = clean_text(str(row["name"]))
        link = str(row["link"]).strip()

        name_words = set(name.split())

        if len(message_words & name_words) >= max(2, len(name_words)//2):

            if link.startswith("www."):
                link = "https://" + link

            if link.startswith("http"):
                return link

    return None

# ---------------- ML PREDICTION ----------------
def predict_message(message):

    msg_clean = clean_text(message)

    tfidf = vectorizer.transform([msg_clean])

    prediction = model.predict(tfidf)[0]

    probabilities = model.predict_proba(tfidf)[0]

    if prediction == 1:
        return "FAKE", round(probabilities[1],3)
    else:
        return "GENUINE", round(probabilities[0],3)

# ---------------- HOME PAGE ----------------
@app.route("/", methods=["GET","POST"])
def home():

    result = None
    confidence = None
    message = ""
    link = None

    if request.method == "POST":

        message = request.form["message"]

        # Step 1: Check scholarship message
        if not is_scholarship_message(message):

            result = "NOT A SCHOLARSHIP MESSAGE"
            confidence = 0
            link = None

        else:

            # Step 2: Check database
            found_link = find_scholarship_link(message)

            if found_link:

                result = "GENUINE"
                confidence = 0.95
                link = found_link

            else:

                # Step 3: Rule based detection
                result, confidence = rule_based_detection(message)

                if result is None:

                    # Step 4: Machine learning
                    result, confidence = predict_message(message)

                link = None

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message,
        link=link
    )

# ---------------- FEEDBACK ----------------
@app.route("/feedback", methods=["POST"])
def feedback():

    link = request.form.get("link","")

    entry = {

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": request.form.get("message"),
        "predicted_label": request.form.get("predicted_label"),
        "confidence": request.form.get("confidence"),
        "link": link,
        "satisfaction": request.form.get("satisfaction"),
        "rating": request.form.get("rating")

    }

    pd.DataFrame([entry]).to_csv(
        feedback_file,
        mode="a",
        header=False,
        index=False
    )

    return redirect(url_for("home"))

# ---------------- ADMIN PAGE ----------------
@app.route("/admin")
def admin():

    df = pd.read_csv(feedback_file)

    if not df.empty:

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        avg_rating = round(df["rating"].mean(),2)

        satisfaction_rate = round(
            (df["satisfaction"] == "Yes").mean() * 100,
            2
        )

        df["scholarship_link"] = df["link"].apply(
            lambda x: x if pd.notna(x) and x.strip() else None
        )

    else:

        avg_rating = 0
        satisfaction_rate = 0

    records = df.to_dict(orient="records")

    return render_template(
        "admin.html",
        records=records,
        avg_rating=avg_rating,
        satisfaction_rate=satisfaction_rate
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)