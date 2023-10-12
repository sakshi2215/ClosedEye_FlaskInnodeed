import numpy as np
import cv2 as cv
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

# Constants
FONTS = cv.FONT_HERSHEY_COMPLEX
# Face boundary indices
# Face boundary indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Left eye indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Right eye indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh

model = load_model('my_model.h5')
app = Flask(__name__)

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'photo' not in request.files:
        return render_template('upload.html', result="No file selected.")

    photo = request.files['photo']
    if photo.filename == '':
        return render_template('upload.html', result="No file selected.")

    image = cv.imdecode(np.fromstring(photo.read(), np.uint8), cv.IMREAD_COLOR)
    if image is None:
        return render_template('upload.html', result="Failed to load image.")

    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # Convert image from RGB to BGR
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            mesh_coords = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in results.multi_face_landmarks[0].landmark]

            # Extract the bounding rectangle for the left and right eye
            left_eye_rect = cv.boundingRect(np.array([mesh_coords[p] for p in LEFT_EYE]))
            right_eye_rect = cv.boundingRect(np.array([mesh_coords[p] for p in RIGHT_EYE]))

             # Scaling factors to increase the height and width of bounding boxes
            height_scale = 1.5
            width_scale = 1.3

            # Calculate the new height and width of the bounding boxes
            new_left_eye_height = int(left_eye_rect[3] * height_scale)
            new_left_eye_width = int(left_eye_rect[2] * width_scale)
            new_right_eye_height = int(right_eye_rect[3] * height_scale)
            new_right_eye_width = int(right_eye_rect[2] * width_scale)

            # Calculate the new (x, y) positions for the bounding boxes
            new_left_eye_x = max(0, left_eye_rect[0] - int((new_left_eye_width - left_eye_rect[2]) / 2))
            new_left_eye_y = max(0, left_eye_rect[1] - int((new_left_eye_height - left_eye_rect[3]) / 2))
            new_right_eye_x = max(0, right_eye_rect[0] - int((new_right_eye_width - right_eye_rect[2]) / 2))
            new_right_eye_y = max(0, right_eye_rect[1] - int((new_right_eye_height - right_eye_rect[3]) / 2))

            # Adjust the height and width of the bounding boxes
            new_left_eye_height = min(image.shape[0] - new_left_eye_y, new_left_eye_height)
            new_left_eye_width = min(image.shape[1] - new_left_eye_x, new_left_eye_width)
            new_right_eye_height = min(image.shape[0] - new_right_eye_y, new_right_eye_height)
            new_right_eye_width = min(image.shape[1] - new_right_eye_x, new_right_eye_width)

            # Crop the left and right eye regions from the original image using the new bounding box sizes
            left_eye_image = image[new_left_eye_y:new_left_eye_y + new_left_eye_height,
                                   new_left_eye_x:new_left_eye_x + new_left_eye_width]
            right_eye_image = image[new_right_eye_y:new_right_eye_y + new_right_eye_height,
                                    new_right_eye_x:new_right_eye_x + new_right_eye_width]

            # Resize the left and right eye images to (224, 224) and preprocess
            final_left_eye = cv.resize(left_eye_image, (224, 224))
            final_left_eye = np.expand_dims(final_left_eye, axis=0)
            final_left_eye = final_left_eye / 255.0

            final_right_eye = cv.resize(right_eye_image, (224, 224))
            final_right_eye = np.expand_dims(final_right_eye, axis=0)
            final_right_eye = final_right_eye / 255.0

            # Make the predictions for the left and right eyes
            left_eye_prediction = model.predict(final_left_eye)
            right_eye_prediction = model.predict(final_right_eye)

            # Determine the result based on the predictions
            if left_eye_prediction[0] > 0.5 and right_eye_prediction[0] > 0.5:
                result = " Open Eyes"
            elif left_eye_prediction[0] > 0.5 or right_eye_prediction[0] > 0.5:
                result = " Open Eyes"
            else:
                result = "Closed Eyes"

            # Draw the adjusted bounding rectangles on the original image for visualization
            cv.rectangle(image, (new_left_eye_x, new_left_eye_y),
                          (new_left_eye_x + new_left_eye_width, new_left_eye_y + new_left_eye_height), (0, 255, 0), 2)
            cv.rectangle(image, (new_right_eye_x, new_right_eye_y),
                          (new_right_eye_x + new_right_eye_width, new_right_eye_y + new_right_eye_height), (0, 255, 0), 2)

          

            # # Draw the result text on the image
            cv.putText(image, result, (10, 30), FONTS, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Save the image with detection result
    result_image_path = "static/result.png"
    cv.imwrite(result_image_path, image)

    return render_template('upload.html', result=result, image_path=result_image_path)

  

if __name__ == '__main__':
    app.run(debug=True)
