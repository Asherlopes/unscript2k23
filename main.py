from flask import Flask,request, url_for, redirect, render_template
import pickle
import joblib
import cv2
import numpy as np
from flask_mysqldb import MySQL


app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'unscript'

mysql = MySQL(app)
#
# model=pickle.load(open('model.pkl','rb'))
def calculate_noise(image):
    # Load the image
    # image = cv2.imread('image', 0)
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the standard deviation of the pixel values
    std_dev = np.std(image)

    return std_dev

def calculate_vignetting(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness values in the center and at the periphery of the image
    center_brightness = np.mean(gray[int(gray.shape[0] / 4): int(gray.shape[0] * 3 / 4), int(gray.shape[1] / 4): int(gray.shape[1] * 3 / 4)])
    periphery_brightness = np.mean(np.concatenate((gray[:, 0: int(gray.shape[1] / 4)], gray[:, int(gray.shape[1] * 3 / 4):]), axis=1))

    # Calculate the amount of vignetting as the difference between the center and periphery brightness values
    vignetting = center_brightness - periphery_brightness

    return vignetting

def calculate_exposure(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist = hist / (gray.shape[0] * gray.shape[1])

    # Calculate the entropy of the histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy

def calculate_sharpness(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the gradient magnitude
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the standard deviation of the gradient magnitude
    stddev = np.std(gradient_magnitude)

    # The sharpness score can be defined as the inverse of the standard deviation
    sharpness = 1 / stddev

    return sharpness

def calculate_blurriness(image):
    # Load the image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Calculate the gradient magnitude using the Canny edge detection algorithm
    gradient_magnitude = cv2.Canny(img, 50, 150)

    # Return the average value of the gradient magnitude
    return np.mean(gradient_magnitude)



@app.route('/')
def hello_world():
    return render_template("index.html")



@app.route("/check", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image file
        # file = request.files["file"]
        image_file = request.files.get("image")
                # file = request.files['file']
                # Preprocessing and feature extraction of the image
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Read the image
        image = cv2.imread(image_file.filename)
        # Preprocessing
        # image = cv2.resize(image, (224,224))


        # Feature extraction
        noise = calculate_noise(image)
        vignetting = calculate_vignetting(image)
        exposure = calculate_exposure(image)
        sharpness = calculate_sharpness(image)
        blurriness = calculate_blurriness(image_file.filename)
        # Normalization
        features = [noise, vignetting, exposure, sharpness, blurriness]
        features = (features - np.mean(features)) / np.std(features)
        # print(features)
        # Model prediction
        model = joblib.load("model.pkl")
        prediction = model.predict([features])[0]
        # Return the prediction result to the user
        print(prediction)
        #
        # if prediction[0] == 0:
        #    prediction_string = ' Very Good'
        # elif prediction[0] == 1:
        #         prediction_string = 'Good'
        # elif prediction[0] == 2:
        #             prediction_string = 'Bad'
        # else:
        #             prediction_string = 'Poor'
        return render_template("check.html", prediction=prediction)
    return render_template("check.html")

    # return render_template('check.html', prediction_string=prediction_string)
#     return render_template('check.html')




#
#
#
# @app.route('/check',methods=['POST','GET'])
# def predict():
#     int_features=[int(x) for x in request.form.values()]
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)
#
#     if output>str(0.5):
#         return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
#
#
# if __name__ == '__main__':
#     app.run(debug=True)









# from flask import Flask, render_template, request, redirect, url_for, flash, session, request, jsonify
# from flask_mysqldb import MySQL
# import os
# from sklearn.preprocessing import MinMaxScaler
#
# import joblib
#
# import pickle
#
#
# import numpy as np
# import cv2
#
# #
# # from sklearn.externals import joblib
# # #
# model = joblib.load('model.pkl')
#
# # Load the ML model from the pickle file
#
#
#
#
# def calculate_noise(image):
#     # Load the image
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Calculate the standard deviation of the pixel values
#     std_dev = np.std(gray)
#
#     return std_dev
#
# def calculate_vignetting(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Calculate the average brightness values in the center and at the periphery of the image
#     center_brightness = np.mean(gray[int(gray.shape[0] / 4): int(gray.shape[0] * 3 / 4), int(gray.shape[1] / 4): int(gray.shape[1] * 3 / 4)])
#     periphery_brightness = np.mean(np.concatenate((gray[:, 0: int(gray.shape[1] / 4)], gray[:, int(gray.shape[1] * 3 / 4):]), axis=1))
#
#     # Calculate the amount of vignetting as the difference between the center and periphery brightness values
#     vignetting = center_brightness - periphery_brightness
#
#     return vignetting
#
# def calculate_exposure_accuracy(image):
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Calculate the histogram of the image
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#
#     # Normalize the histogram
#     hist = hist / (gray.shape[0] * gray.shape[1])
#
#     # Calculate the entropy of the histogram
#     entropy = -np.sum(hist * np.log2(hist + 1e-10))
#
#     return entropy
#
# def calculate_sharpness(image):
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Compute the gradient magnitude
#     gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
#
#     # Compute the standard deviation of the gradient magnitude
#     stddev = np.std(gradient_magnitude)
#
#     # The sharpness score can be defined as the inverse of the standard deviation
#     sharpness = 1 / stddev
#
#     return sharpness
#
#
#
#
#
# app = Flask(__name__)
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'unscript'
#
# mysql = MySQL(app)
#
# @app.route("/")  # this sets the route to this page
# def home():
#     return render_template("index.html")
# # Flask application file
#
#
#
#
#
# #
# # main file
# # #
# @app.route('/check', methods=['GET', 'POST'])
# def check():
#     if request.method == 'POST':
#         image_file = request.files.get("image")
#         # file = request.files['file']
#         # Preprocessing and feature extraction of the image
#         image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#         # Calculate the sharpness, noise, clearness, and blurriness features here
#
#         # Extract features from the image
#         sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
#         noise = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
#         noise = cv2.Laplacian(noise, cv2.CV_64F).var()
#         clearness = cv2.Laplacian(noise, cv2.CV_64F).var()
#         blurness = cv2.Laplacian(noise, cv2.CV_64F).var()
#
#         vignating = calculate_vignetting(image)
#         # Stack the features into a single array
#         features = np.array([sharpness, noise, clearness, blurness,vignating]).reshape(1, -1)
#
#         scaler = MinMaxScaler()
#
#         # fit the scaler to the features
#         scaler.fit(features)
#
#         # transform the features to the normalized scale
#         normalized_features = scaler.transform(features)
#
#         # load the logistic regression model from the .pkl file
#         model = joblib.load('model.pkl')
#
#         # make predictions using the normalized features
#         prediction = model.predict(normalized_features)
#
#         # map the predicted class label to a string
#         if prediction[0] == 0:
#             prediction_string = ' Very Good'
#         elif prediction[0] == 1:
#             prediction_string = 'Good'
#         elif prediction[0] == 2:
#             prediction_string = 'Bad'
#         else:
#             prediction_string = 'Poor'
#
#         # return the prediction string to the user
#         # return prediction_string
#
#         return render_template('check.html', prediction_string=prediction_string)
#     return render_template('check.html')
#
#
#
#
#
#
#


@app.route("/feedback",methods=["POST","GET"])  # this sets the route to this page
def feedback():
    if request.method == "POST":
        details = request.form
        Name = details['name']
        email = details['email']
        feedback = details['feedback']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO feedback(name,email, feedback) VALUES (%s, %s, %s)", (Name, email, feedback))
        mysql.connection.commit()
        cur.close()
        # return 'success'

        return '<script>alert("Feedback Submitted");window.location="/"</script>'


    return render_template("feedback.html")

#
# #
# # def print_hi(name):
# #     # Use a breakpoint in the code line below to debug your script.
# #     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
# #
# #
#
#
#
if __name__ == "__main__":
    app.run(debug=True)