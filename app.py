# from flask import Flask, render_template, request
# import numpy as np
# import cv2
# from keras.models import load_model
# import webbrowser

# # Initialize Flask app
# app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# # Global variables
# info = {}
# haarcascade = "haarcascade_frontalface_default.xml"
# label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise']

# # Load the face detection model and emotion classification model
# try:
#     print("+" * 50, "Loading Models")
    
#     # Emotion classification model
#     model = load_model('model.h5')
    
#     # Load DNN-based face detection model
#     modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
#     configFile = "deploy.prototxt.txt"
#     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# except Exception as e:
#     print(f"Error loading model or cascade: {str(e)}")

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Choose singer route
# @app.route('/choose_singer', methods=["POST"])
# def choose_singer():
#     try:
#         info['language'] = request.form['language']
#         print(info)
#         return render_template('choose_singer.html', data=info['language'])
#     except Exception as e:
#         return f"Error processing form data: {str(e)}", 500

# # Emotion detection route
# @app.route('/emotion_detect', methods=["POST"])
# def emotion_detect():
#     try:
#         info['singer'] = request.form['singer']

#         # Initialize webcam
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             return "Error: Could not access the camera. Please ensure the webcam is connected and try again.", 500

#         found = False
#         roi = None

#         # Capture frames until a face is detected using DNN
#         while not found:
#             ret, frm = cap.read()
#             if not ret:
#                 return "Error: Failed to read from the camera.", 500

#             h, w = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#             net.setInput(blob)
#             detections = net.forward()

#             # Loop over detected faces
#             for i in range(detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]

#                 if confidence > 0.80:  # Confidence threshold
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     x1, y1, x2, y2 = box.astype("int")
#                     roi = frm[y1:y2, x1:x2]  # Extract face

#                     cv2.imwrite("static/face.jpg", roi)  # Save captured face image
#                     found = True
#                     break

#         cap.release()

#         # Ensure a face region was captured
#         if roi is None:
#             return "No face detected. Please try again.", 400

#         # Preprocess the face region
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         roi_resized = cv2.resize(gray, (48, 48))
#         roi_normalized = roi_resized / 255.0
#         roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

#         # Predict the emotion
#         prediction = model.predict(roi_reshaped)
#         prediction = np.argmax(prediction)
#         prediction_label = label_map[prediction]

#         # Check if the emotion is "anger" or "sad"
#         if  prediction_label == "sad" or prediction_label == "anger" :
#             # Ask the user if they want neutral or happy music
#             return render_template("ask_music_preference.html", data=prediction_label)

#         # Generate YouTube search link for other emotions
#         link = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction_label}+{info['language']}+song"
#         webbrowser.open(link)

#         return render_template("emotion_detect.html", data=prediction_label, link=link)
#     except Exception as e:
#         return f"Error during emotion detection: {str(e)}", 500

# # Recommend music based on preference
# @app.route('/recommend_music', methods=["POST"])
# def recommend_music():
#     try:
#         preference = request.form['preference']
#         link = f"https://www.youtube.com/results?search_query={info['singer']}+{preference}+{info['language']}+song"
#         webbrowser.open(link)
#         return render_template("music_recommendation.html", data=preference, link=link)
#     except Exception as e:
#         return f"Error during music recommendation: {str(e)}", 500
# @app.route('/trainer')

# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)

'''
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
import subprocess
# Initialize Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Global variables
info = {}
label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise']

# Load the face detection model and emotion classification model
try:
    print("+" * 50, "Loading Models")

    # Emotion classification model
    model = load_model('model.h5')

    # Load DNN-based face detection model
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
except Exception as e:
    print(f"Error loading model or cascade: {str(e)}")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Trainer page route
@app.route('/trainer')
def trainer():
    return render_template('trainer.html')

# for collect data button
@app.route('/collect_data', methods=["POST"])
def collect_data():
    try:
        emotion = request.form['emotion'].strip().lower()

        # Run the external data collection script
        result = subprocess.run(
            ['python', 'data_collection.py', emotion],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return f"<h3>Error during data collection:</h3><pre>{result.stderr}</pre>", 500

        return f"<h3>✅ Data collected successfully for emotion: '{emotion}'</h3><pre>{result.stdout}</pre>"

    except Exception as e:
        return f"Error: {str(e)}", 500
    
# for tain data button
@app.route('/train_model', methods=["POST"])
def train_model():
    try:
        result = subprocess.run(
            ['python', 'data_train.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return f"<h3>Error during training:</h3><pre>{result.stderr}</pre>", 500

        return "<h3>✅ Model trained successfully. Output saved as <code>emotion_model.h5</code>.</h3><pre>{}</pre>".format(result.stdout)

    except Exception as e:
        return f"Error: {str(e)}", 500

# Choose singer route
@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    try:
        info['language'] = request.form['language']
        return render_template('choose_singer.html', data=info['language'])
    except Exception as e:
        return f"Error processing form data: {str(e)}", 500
    

    # import subprocess


# Emotion detection route
@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    try:
        info['singer'] = request.form['singer']

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not access the camera.", 500

        found = False
        roi = None

        while not found:
            ret, frm = cap.read()
            if not ret:
                return "Error: Failed to read from camera.", 500

            h, w = frm.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.80:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    roi = frm[y1:y2, x1:x2]
                    cv2.imwrite("static/face.jpg", roi)
                    found = True
                    break

        cap.release()

        if roi is None:
            return "No face detected. Please try again.", 400

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        prediction = model.predict(roi_reshaped)
        prediction = np.argmax(prediction)
        prediction_label = label_map[prediction]

        if prediction_label in ["sad", "anger"]:
            return render_template("ask_music_preference.html", data=prediction_label)

        link = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction_label}+{info['language']}+song"
        webbrowser.open(link)
        return render_template("emotion_detect.html", data=prediction_label, link=link)

    except Exception as e:
        return f"Error during emotion detection: {str(e)}", 500

# Recommend music based on preference
@app.route('/recommend_music', methods=["POST"])
def recommend_music():
    try:
        preference = request.form['preference']
        link = f"https://www.youtube.com/results?search_query={info['singer']}+{preference}+{info['language']}+song"
        webbrowser.open(link)
        return render_template("music_recommendation.html", data=preference, link=link)
    except Exception as e:
        return f"Error during music recommendation: {str(e)}", 500

# Run the app
if __name__ == "__main__":
     app.run(debug=True)

'''
'''

from flask import Flask, render_template, request, redirect, url_for, flash, session
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib, ssl
from email.message import EmailMessage
import numpy as np
import cv2
from keras.models import load_model
import subprocess
import webbrowser

app = Flask(__name__)
app.secret_key = 'supersecretkey'
s = URLSafeTimedSerializer(app.secret_key)

# MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.emoify
users_col = db.users
trainers_col = db.trainers

# Email setup
EMAIL_ADDRESS = "youremail@gmail.com"
EMAIL_PASSWORD = "your_app_password"

# Load models
try:
    model = load_model('model.h5')
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise']
except Exception as e:
    print(f"Model loading error: {e}")

def send_email(to_email, subject, content):
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# ------------------- ROUTES --------------------

@app.route('/')
def home():
    # Default landing page
    if 'user' in session:
        return redirect(url_for('user_dashboard'))
    elif 'trainer' in session:
        return redirect(url_for('trainer_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        role = request.form['role']

        collection = users_col if role == 'user' else trainers_col
        user = collection.find_one({'email': email})

        if not user or not check_password_hash(user['password'], password):
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

        if role == 'user' and not user.get('verified'):
            flash('Email not verified.', 'warning')
            return redirect(url_for('login'))

        session.clear()
        session[role] = email
        return redirect(url_for('user_dashboard') if role == 'user' else url_for('trainer_dashboard'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']

        if users_col.find_one({'email': email}):
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        token = s.dumps(email, salt='email-confirm')
        link = url_for('confirm_email', token=token, _external=True)

        users_col.insert_one({
            'email': email,
            'password': hashed_pw,
            'verified': False
        })

        send_email(email, "Verify Your Email", f"Click to verify your account: {link}")
        flash('Verification link sent to your email.', 'info')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/confirm_email/<token>')
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)
        users_col.update_one({'email': email}, {'$set': {'verified': True}})
        flash('Email verified. Please log in.', 'success')
    except (SignatureExpired, BadSignature):
        flash('Verification link expired or invalid.', 'danger')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ----------------- DASHBOARDS ------------------

@app.route('/user_dashboard')
def user_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/trainer_dashboard')
def trainer_dashboard():
    if 'trainer' not in session:
        return redirect(url_for('login'))
    return render_template("trainer.html")

# ----------------- EMOTION & MUSIC --------------

@app.route('/choose_singer', methods=['POST'])
def choose_singer():
    info['language'] = request.form['language']
    return render_template('choose_singer.html', data=info['language'])

@app.route('/emotion_detect', methods=['POST'])
def emotion_detect():
    info['singer'] = request.form['singer']
    cap = cv2.VideoCapture(0)
    found = False
    while not found:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite("static/face.jpg", roi)
                found = True
                break

    cap.release()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(gray, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    prediction = model.predict(roi_reshaped)
    emotion = label_map[np.argmax(prediction)]

    if emotion in ["sad", "anger"]:
        return render_template("ask_music_preference.html", data=emotion)

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{emotion}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("emotion_detect.html", data=emotion, link=link)

@app.route('/recommend_music', methods=['POST'])
def recommend_music():
    preference = request.form['preference']
    link = f"https://www.youtube.com/results?search_query={info['singer']}+{preference}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("music_recommendation.html", data=preference, link=link)

# ------------------ TRAINER TOOLS ------------------

@app.route('/collect_data', methods=['POST'])
def collect_data():
    emotion = request.form['emotion'].strip().lower()
    result = subprocess.run(['python', 'data_collection.py', emotion], capture_output=True, text=True)
    return f"<h3>Data collected for '{emotion}'</h3><pre>{result.stdout}</pre>"

@app.route('/train_model', methods=['POST'])
def train_model():
    result = subprocess.run(['python', 'data_train.py'], capture_output=True, text=True)
    return f"<h3>Model trained.</h3><pre>{result.stdout}</pre>"

# ----------------- MAIN ------------------
if __name__ == '__main__':
    info = {}
    app.run(debug=True)
# '''
# from flask import Flask, render_template, request, redirect, url_for, flash, session
# from pymongo import MongoClient
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# app.secret_key = 'supersecretkey'

# # MongoDB Connection
# client = MongoClient('mongodb://localhost:27017/')
# db = client.emoify
# users_col = db.users
# trainers_col = db.trainers

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email', '').lower()
#         password = request.form.get('password', '')

#         user = users_col.find_one({'email': email})
        
#         if not user:
#             flash("User does not exist. Please sign up.", "danger")
#             return redirect(url_for('login'))
        
#         if not check_password_hash(user['password'], password):
#             flash("Invalid username or password.", "danger")
#             return redirect(url_for('login'))

#         session.clear()
#         session['user'] = email
#         flash("Login successful!", "success")
#         return redirect(url_for('user_dashboard'))  # Redirect to index.html
        
#     return render_template('login.html')

# @app.route('/user_dashboard')
# def user_dashboard():
#     if 'user' not in session:
#         return redirect(url_for('login'))
#     return render_template('index.html')

# # Run the application
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
from keras.models import load_model
import subprocess
import webbrowser

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client.emoify
users_col = db.users

# Load Emotion Detection Model
try:
    model = load_model('model.h5')
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    label_map = ['anger', 'neutral', 'fear', 'happy', 'sad', 'surprise']
except Exception as e:
    print(f"Model loading error: {e}")

info = {}

# -------------------- ROUTES --------------------

@app.route('/')
def home():
    return redirect(url_for('login'))

# User Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        role = request.form.get('role', 'user')

        if role == 'trainer':
            flash('Trainer accounts cannot be created via signup.', 'danger')
            return redirect(url_for('signup'))

        if users_col.find_one({'email': email}):
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        users_col.insert_one({
            'email': email,
            'password': hashed_pw,
            'verified': True
        })

        flash('User account created. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']

        user = users_col.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session.clear()
            session['user'] = email
            return redirect(url_for('index'))
        else:
            flash('Invalid user credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Trainer Login (separate route)
@app.route('/trainer-login', methods=['POST'])
def trainer_login():
    username = request.form['username']
    password = request.form['password']

    if username == 'admin' and password == 'admin':
        session.clear()
        session['trainer'] = username
        return redirect(url_for('trainer'))
    else:
        flash('Username or password is wrong.', 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/trainer')
def trainer():
    if 'trainer' not in session:
        flash('Please log in as trainer to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('trainer.html', message='')


# -------------------- EMOTION & MUSIC --------------------

@app.route('/choose_singer', methods=['POST'])
def choose_singer():
    info['language'] = request.form['language']
    return render_template('choose_singer.html', data=info['language'])

@app.route('/emotion_detect', methods=['POST'])
def emotion_detect():
    info['singer'] = request.form['singer']
    cap = cv2.VideoCapture(0)
    found = False
    roi = None
    while not found:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite("static/face.jpg", roi)
                found = True
                break

    cap.release()

    if roi is None:
        flash("Face not detected. Please try again.", "danger")
        return redirect(url_for('index'))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(gray, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    prediction = model.predict(roi_reshaped)
    emotion = label_map[np.argmax(prediction)]

    if emotion in ["sad", "anger"]:
        return render_template("ask_music_preference.html", data=emotion)

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{emotion}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("emotion_detect.html", data=emotion, link=link)

@app.route('/recommend_music', methods=['POST'])
def recommend_music():
    preference = request.form['preference']
    link = f"https://www.youtube.com/results?search_query={info['singer']}+{preference}+{info['language']}+song"
    webbrowser.open(link)
    return render_template("music_recommendation.html", data=preference, link=link)

# -------------------- TRAINER TOOLS --------------------

@app.route('/collect_data', methods=['POST'])
def collect_data():
    if 'trainer' not in session:
        flash('Please login as trainer to access this.', 'warning')
        return redirect(url_for('login'))

    emotion = request.form['emotion'].strip().lower()
    result = subprocess.run(['python', 'data_collection.py', emotion], capture_output=True, text=True)
    message = f"Data collected for '{emotion}':\n{result.stdout}"
    return render_template('trainer.html', message=message)

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'trainer' not in session:
        flash('Please login as trainer to access this.', 'warning')
        return redirect(url_for('login'))

    result = subprocess.run(['python', 'data_train.py'], capture_output=True, text=True)
    message = f"Model training completed:\n{result.stdout}"
    return render_template('trainer.html', message=message)

# -------------------- RUN --------------------

if __name__ == '__main__':
    app.run(debug=True)

