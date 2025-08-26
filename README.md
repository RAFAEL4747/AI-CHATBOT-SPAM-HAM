# AI-CHATBOT-SPAM-HAM
To build a machine learning model that can classify emails as spam or ham (non-spam) based on their content using Support Vector Machine (SVM).
Title: Spam Email Detection using SVM

Objective:
To build a machine learning model that can classify emails as spam or ham (non-spam) based on their content using Support Vector Machine (SVM).

Dataset:

Embedded dataset of 50 email messages with labels "ham" or "spam".

Converted labels to binary: ham=0, spam=1.

Methodology:

Split dataset into training (80%) and testing (20%) sets.

Used a pipeline with:

CountVectorizer for tokenizing text.

TfidfTransformer to weigh word importance.

LinearSVC classifier for prediction.

Trained the model on the training set.

Evaluated on the test set using accuracy and classification report.

Results:

Provides a detailed classification report with precision, recall, F1-score, and overall accuracy.

Allows testing of custom email inputs for real-time spam detection.

Conclusion:
The SVM-based email classifier effectively distinguishes spam from ham using text content and can be scaled with larger datasets for improved accuracy.
