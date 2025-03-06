from flask import Flask, render_template, redirect, url_for, request, Response
import cv2
import subprocess
from hotspot_analysis import run_hotspot_analysis

app = Flask(__name__)

# Define the highlightFace function
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes

# Function to generate video frames
def generate_frames():
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)  # Use 0 for webcam

    if not video.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = video.read()
        if not success:
            print("Error: Could not read frame.")
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")

        menCount = 0
        womenCount = 0
        totalCount = len(faceBoxes)

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-20):min(faceBox[3]+20, frame.shape[0]-1),
                         max(0, faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            if gender == 'Male':
                menCount += 1
                color = (255, 0, 0)  # Blue for men
            else:
                womenCount += 1
                color = (255, 255, 255)  # White for women

            if womenCount == 1 and totalCount == 1:
                color = (0, 255, 255)  # Yellow for alone woman
                cv2.putText(resultImg, 'Woman is alone', (faceBox[0], faceBox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), color, int(round(frame.shape[0]/150)), 8)
            cv2.putText(resultImg, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        if womenCount > 0 and menCount / womenCount >= 3:
            cv2.putText(resultImg, 'Gender Imbalance Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(resultImg, f'Total: {totalCount} | Men: {menCount} | Women: {womenCount}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', resultImg)
        if not ret:
            print("Error: Could not encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()

# Route to the login page
@app.route('/')
def login():
    return render_template('login.html')

# Route that handles login
@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']

    # Simple username/password check (replace with proper authentication)
    if username == 'admin' and password == 'password':
        return redirect(url_for('dashboard'))
    else:
        error = "Invalid credentials"
        return render_template('login.html', error=error)

# Route to the dashboard after successful login
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route that runs hotspot analysis
@app.route('/run_hotspot_analysis', methods=['POST'])
def run_hotspot_analysis_route():
    map_html = run_hotspot_analysis()  # Call the function directly
    return render_template('hotspot_map.html', map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
