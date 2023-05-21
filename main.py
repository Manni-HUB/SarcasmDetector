import json
import sys
from textblob import TextBlob
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QMovie
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer,QSize
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication, QMainWindow, QLineEdit, QPushButton




nltk.download('wordnet')
nltk.download('sentiwordnet')


data = []  # List to store the collected data
X = []
y = []
i = 0 
# Open the file and read line by line
with open('./Sarcasm_Headlines_Dataset.json', 'r') as file:
    for line in file:
        try:
            tweet = json.loads(line)
            # Extract the relevant fields (headline and is_sarcastic)
            headline = tweet['headline']
            is_sarcastic = tweet['is_sarcastic']
            
            # Add the extracted data as a tuple to the data list
            data.append((headline, is_sarcastic))
        except json.JSONDecodeError:
            # Handle JSON decoding errors
            print('JSON decoding error:', line)
            continue

        i += 1 
        if i > 8000:
            break 

# Construct feature vectors using proposed features
for headline, is_sarcastic in data:
    # Create a TextBlob object what it does is to enables you to leverage the functionalities of the TextBlob library to analyze and process the text. 
    # such as noun phrase extraction, sentiment analysis, and more.
    blob = TextBlob(headline)
    
    # Get the sentiment polarity means whether the sentence is positive or not 
    polarity = blob.sentiment.polarity
    
    # Perform common-sense knowledge expansion using SentiWordNet and calculates the expanded polarity by considering the sentiment scores of individual words
    expanded_polarity = polarity
    for word in blob.words:
        pos_score = 0
        neg_score = 0

        #SentiWordNet assigns sentiment scores to words based on their meanings.
        synsets = list(swn.senti_synsets(word))
        if synsets:
            for synset in synsets:
                pos_score += synset.pos_score()
                neg_score += synset.neg_score()
        
        # Adjust the polarity based on the expanded sentiment scores
        # The expanded_polarity variable represents the sentiment polarity adjusted by considering the sentiment scores of individual words.
        expanded_polarity += pos_score - neg_score
    
    # Split the headline into sentences
    sentences = sent_tokenize(headline)
    
    # Perform sentiment analysis on each sentence
    sentence_sentiments = []
    for sentence in sentences:
        sentence_blob = TextBlob(sentence)
        sentence_polarity = sentence_blob.sentiment.polarity
        sentence_sentiments.append(sentence_polarity)
    
    # Construct the feature vector
    feature_vector = [polarity, expanded_polarity] + sentence_sentiments

    X.append(feature_vector)
    # append The label (is_sarcasti) to the y list  
    y.append(is_sarcastic)

# Determine the maximum length of sublists in X
max_length = max(len(row) for row in X)

# Pad sublists with zeros to make them all have the same length
X_padded = [row + [0] * (max_length - len(row)) for row in X]

# Convert X to a NumPy array
X = np.asarray(X_padded, dtype=np.float32)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier using the proposed features upport Vector Machine (SVM)
svm_proposed = SVC(kernel='linear', C=1.0)
svm_proposed.fit(X_train, y_train)

# Make predictions on the testing set using the proposed features
y_pred_proposed = svm_proposed.predict(X_test)

# Calculate the accuracy of the classifier using the proposed features

#Accuracy is defined as the ratio of the correctly predicted samples to the total number of samples.
accuracy_proposed = accuracy_score(y_test, y_pred_proposed)

#Precision is defined as the ratio of true positive predictions to the sum of true positive and false positive predictions.
precision_proposed = precision_score(y_test, y_pred_proposed,zero_division=0)

#Recall is defined as the ratio of true positive predictions to the sum of true positive and false negative predictions.
recall_proposed = recall_score(y_test, y_pred_proposed,zero_division=0)

#The F1 score is defined as the harmonic mean of precision and recall, and it provides a balanced assessment of a model's performance.
f1_score_proposed = f1_score(y_test, y_pred_proposed,zero_division=0)

print('Accuracy (Proposed Features):', accuracy_proposed)
print('Precision (Proposed Features):', precision_proposed)
print('Recall (Proposed Features):', recall_proposed)
print('F1 Score (Proposed Features):', f1_score_proposed)
# Extract N-gram features
vectorizer = CountVectorizer(ngram_range=(1, 3))
X_ngram = vectorizer.fit_transform([headline for headline, _ in data]).toarray()


# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)  # Choose the desired number of components
X_ngram_reduced = pca.fit_transform(X_ngram)

# Split the reduced N-gram features dataset into training and testing sets
X_ngram_reduced_train, X_ngram_reduced_test, y_train, y_test = train_test_split(X_ngram_reduced, y, test_size=0.2, random_state=42)

# Train an SVM classifier using the reduced N-gram features
svm_ngram_reduced = SVC(kernel='linear', C=1.0)
svm_ngram_reduced.fit(X_ngram_reduced_train, y_train)

# Make predictions on the testing set using the reduced N-gram features
y_pred_ngram_reduced = svm_ngram_reduced.predict(X_ngram_reduced_test)

# Calculate the accuracy of the classifier using the reduced N-gram features
accuracy_ngram_reduced = accuracy_score(y_test, y_pred_ngram_reduced)
precision_ngram = precision_score(y_test, y_pred_ngram_reduced)
recall_ngram = recall_score(y_test, y_pred_ngram_reduced)
f1_score_ngram = f1_score(y_test, y_pred_ngram_reduced)


print('Accuracy (Reduced N-gram Features):', accuracy_ngram_reduced)
print('Precision (N-gram Features):', precision_ngram)
print('Recall (N-gram Features):', recall_ngram)
print('F1 Score (N-gram Features):', f1_score_ngram)


def process_user_query(user_tweet):
        # User-entered tweet
 #   user_tweet = input("enter the tweet")

    # Extract features for the user's tweet
    user_blob = TextBlob(user_tweet)
    user_polarity = user_blob.sentiment.polarity
    user_expanded_polarity = user_polarity
    for word in user_blob.words:
        pos_score = 0
        neg_score = 0
        synsets = list(swn.senti_synsets(word))
        if synsets:
            for synset in synsets:
                pos_score += synset.pos_score()
                neg_score += synset.neg_score()
        user_expanded_polarity += pos_score - neg_score

    user_sentences = sent_tokenize(user_tweet)
    user_sentence_sentiments = []
    for sentence in user_sentences:
        sentence_blob = TextBlob(sentence)
        sentence_polarity = sentence_blob.sentiment.polarity
        user_sentence_sentiments.append(sentence_polarity)

    user_feature_vector = [user_polarity, user_expanded_polarity] + user_sentence_sentiments

    # Pad user feature vector with zeros to match the shape of the training data
    user_feature_vector_padded = user_feature_vector + [0] * (max_length - len(user_feature_vector))
    user_feature_vector_np = np.asarray(user_feature_vector_padded, dtype=np.float32)

    # Predict sarcasm using the proposed features
    user_prediction_proposed = svm_proposed.predict(user_feature_vector_np.reshape(1, -1))

    # Extract N-gram features for the user's tweet
    user_ngram_features = vectorizer.transform([user_tweet]).toarray()

    # Apply PCA to reduce dimensionality of the user's N-gram features
    user_ngram_reduced = pca.transform(user_ngram_features)

    # Predict sarcasm using the reduced N-gram features
    user_prediction_ngram_reduced = svm_ngram_reduced.predict(user_ngram_reduced)

    # Print the predictions
    print("Proposed Features Prediction:", user_prediction_proposed[0])
    print("Reduced N-gram Features Prediction:", user_prediction_ngram_reduced[0])


    return user_prediction_ngram_reduced[0] 


# Function to process user input and display the result
# Result Window class
class ResultWindow(QDialog):
    def __init__(self, sarcasm_detected):
        super().__init__()

        self.setWindowTitle("Result")
        self.setGeometry(400, 200, 400, 300)  # Adjusted window position and size

        layout = QVBoxLayout()

        # Display result text
        result_label = QLabel()
        result_label.setFont(QFont("Arial", 16, QFont.Bold))
        result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(result_label)

        # Display GIF image
        gif_label = QLabel()
        if sarcasm_detected:
            movie = QMovie("sarcasm.gif")  # Replace with your GIF file path
        else:
            movie = QMovie("not_sarcastic.gif")  # Replace with your GIF file path
        movie.setScaledSize(QSize(200, 200))
        gif_label.setMovie(movie)
        gif_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(gif_label)

        self.setLayout(layout)

        # Set the result and close the window after a certain time interval
        if sarcasm_detected:
            result_label.setText('<html><body><span style="font-size:18pt; font-weight:bold; color:red;">This tweet is sarcastic</span></body></html>')
        else:
            result_label.setText('<html><body><span style="font-size:18pt; font-weight:bold; color:green;">This tweet is not sarcastic</span></body></html>')
        movie.start()
        QTimer.singleShot(30000, self.close)  # 3000 milliseconds = 3 seconds

# ...

# Function to process user input and display the result
def process_input():
    user_tweet = input_entry.text()  # Get user input from the entry field

    # Perform sarcasm detection based on the user input
    sarcasm_detected = process_user_query(user_tweet)

    # Show the result in a separate window
    result_window = ResultWindow(sarcasm_detected)
    result_window.exec_()
# Create the main application
app = QApplication(sys.argv)

# Create the main window
window = QMainWindow()
window.setWindowTitle("Sarcasm Detection")
window.setGeometry(100, 100, 800, 400)  # Doubled the size of the window

# Set background color
palette = QPalette()
palette.setColor(QPalette.Background, QColor(245, 245, 245))  # Light gray background color
window.setPalette(palette)

# Set background image
background_label = QLabel(window)
background_image = QPixmap("logo1.jpg")  # Replace with your background image file path
background_label.setPixmap(background_image)
background_label.setGeometry(0, 0, 800, 400)  # Doubled the size of the background label
background_label.setScaledContents(True)

# Create a label to display the project name
project_label = QLabel(window)
project_label.setText("Sarcasm Detection")
project_label.setFont(QFont("Arial", 24, QFont.Bold))  # Increased the font size
project_label.setGeometry(200, 80, 400, 60)  # Centered the label
project_label.setAlignment(Qt.AlignCenter)
project_label.setStyleSheet("color: white")  # Set label text color to white

# Create an entry field for the user to enter a tweet
input_entry = QLineEdit(window)
input_entry.setGeometry(250, 160, 300, 40)  # Centered the entry field
input_entry.setPlaceholderText("Enter a tweet")

# Create a button to process the user input
process_button = QPushButton(window)
process_button.setText("Process")
process_button.setGeometry(300, 230, 200, 40) # Centered the button and adjusted size
process_button.setFont(QFont("Arial", 12, QFont.Bold)) # Increased the font size and made it bold
process_button.setStyleSheet("background-color: #00BFFF; color: white") # Set button background color and text color
process_button.clicked.connect(process_input)
window.show()
sys.exit(app.exec_())



