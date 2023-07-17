import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from flask import Flask, render_template, request

model = load_model('my_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
app = Flask(__name__)

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    photo = request.files['photo']
    img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    result = None  # Initialize the result variable
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
            
            final_img = cv2.resize(eyes_roi, (224, 224))
            final_img = np.expand_dims(final_img, axis=0)
            final_img = final_img / 255.0
            prediction = model.predict(final_img)
            
            if prediction[0] > 0.5:
                result = "Open eye detected"
            else:
                result = "Closed eye detected"
                
            cv2.putText(img, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(r"D:\my  projects\INNODEED\eyedetection\static\result.png", img)  # Save the image with detection result
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run()