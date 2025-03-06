import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    menCount = 0
    womenCount = 0
    totalCount = len(faceBoxes)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding)
                     :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        # Count men and women
        if gender == 'Male':
            menCount += 1
            color = (255, 0, 0)  # Blue for men
        else:
            womenCount += 1
            color = (255, 255, 255)  # White for women

        # Trigger 1: Woman is alone
        if womenCount == 1 and totalCount == 1:
            color = (0, 255, 255)  # Yellow for alone woman
            cv2.putText(resultImg, 'Woman is alone', (faceBox[0], faceBox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw the rectangles
        cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), color, int(round(frame.shape[0]/150)), 8)
        cv2.putText(resultImg, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Trigger 2: Men-to-women ratio > 3:1
    if womenCount > 0 and menCount / womenCount >= 3:
        cv2.putText(resultImg, 'Gender Imbalance Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show total person count, men count, and women count
    cv2.putText(resultImg, f'Total: {totalCount} | Men: {menCount} | Women: {womenCount}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gender Detection and Alerts", resultImg)
