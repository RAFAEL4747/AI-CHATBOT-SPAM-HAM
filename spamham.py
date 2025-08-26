import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns

# Dataset
labels = ["ham","spam"]*25
messages = [
    "Hey, are you coming to the party tonight?",
    "Congratulations! You've won a $500 gift card. Click to claim.",
    "Don't forget to submit the assignment by tomorrow.",
    "Win cash prizes now!!! Reply YES to enter.",
    "Can we meet for lunch today?",
    "URGENT! Your account has been compromised. Verify now!",
    "Let's go to the movies this weekend.",
    "Exclusive deal just for you, claim your discount today!",
    "Are you free for a quick call?",
    "You have been selected for a free vacation trip!",
    "Please send me the project report.",
    "Winner! You won $1000. Claim now!",
    "Can you pick up groceries on your way home?",
    "Get rich fast with this one simple trick!",
    "Lunch at 1 pm works for me.",
    "Earn $5000 per week working from home!",
    "Don't forget to water the plants.",
    "Act now! Limited time offer to get free products.",
    "Meeting rescheduled to 3 pm.",
    "You have received a special bonus, claim today!",
    "Let's schedule a call for next week.",
    "Congratulations, you won a free iPhone!",
    "Can you review the presentation slides?",
    "This is your last chance to claim your prize!",
    "Dinner plans for tonight?",
    "Make money online quickly and easily!",
    "Did you complete the homework?",
    "Free entry in a weekly lottery! Click here.",
    "Are we still on for tomorrow?",
    "You've been pre-approved for a credit card offer.",
    "Please confirm your attendance for the meeting.",
    "Earn $1000 daily by working from home!",
    "Pick up your package from the front desk.",
    "Act fast! Limited slots for free training.",
    "Let's go jogging tomorrow morning.",
    "You have been selected for a free gift card.",
    "Can you send me the updated file?",
    "Claim your free vacation now!",
    "I'll call you after the meeting.",
    "Exclusive offer: Buy one get one free!",
    "Happy birthday! Have a great day!",
    "Limited time: Free access to premium content!",
    "Are you attending the workshop next week?",
    "Win a brand new car! Enter now!",
    "Don't forget to bring your ID card tomorrow.",
    "You have won a shopping spree worth $500!",
    "Can you join the team call at 2 pm?",
    "Get free coupons by clicking this link!",
    "Let's meet at the coffee shop.",
    "Immediate action required: Verify your account now!"
]

data = pd.DataFrame({"label": labels, "message": messages})
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
report = metrics.classification_report(y_test, y_pred, target_names=["HAM","SPAM"], output_dict=True)
cm = metrics.confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["HAM","SPAM"], yticklabels=["HAM","SPAM"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# PDF Generation
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Spam Email Classification using SVM", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 6, f"Objective:\nBuild a machine learning model to classify emails as SPAM or HAM using Support Vector Machine (SVM).\n")
pdf.ln(5)

pdf.multi_cell(0, 6, f"Dataset:\nEmbedded dataset of 50 emails with labels 'ham' or 'spam'. Labels converted to binary: ham=0, spam=1.\n")
pdf.ln(5)

pdf.multi_cell(0, 6, f"Methodology:\n- Train/Test split: 80%/20%\n- Pipeline: CountVectorizer -> TFIDF -> LinearSVC\n- Evaluation metrics: Accuracy, Classification Report, Confusion Matrix\n")
pdf.ln(5)

pdf.multi_cell(0, 6, f"Results:\nAccuracy: {accuracy*100:.2f}%\n")
pdf.image("confusion_matrix.png", x=50, w=110)
pdf.ln(5)

pdf.multi_cell(0, 6, "Classification Report Summary:")
for label in ["HAM","SPAM"]:
    pdf.multi_cell(0, 5, f"{label} - Precision: {report[label]['precision']:.2f}, Recall: {report[label]['recall']:.2f}, F1-score: {report[label]['f1-score']:.2f}")

pdf.output("Spam_Classifier_Report.pdf")
print("PDF report generated as 'Spam_Classifier_Report.pdf'")
