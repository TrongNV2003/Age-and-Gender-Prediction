import cv2

# Prepare data input
def highlightFace(net, frame, conf_threshold=0.7):
    # These lines create a copy of the frame to avoid modifying the original and obtain its dimensions.
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Create a blob from the image and run the network
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    
    # Run the network to get the detections
    detections = net.forward()
    
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            startX = int(detections[0,0,i,3] * frameWidth)
            startY = int(detections[0,0,i,4] * frameHeight)
            endX = int(detections[0,0,i,5] * frameWidth)
            endY = int(detections[0,0,i,6] * frameHeight)
            faceBoxes.append([startX,startY,endX,endY])
            cv2.rectangle(frameOpencvDnn, (startX,startY), (endX,endY), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# Load model pre-trained
path = 'C:/Users/Trongdz/Desktop/Age and Gender Detection/models/' 

faceProto= path + "opencv_face_detector.pbtxt"
faceModel= path + "opencv_face_detector_uint8.pb"

ageProto= path + "age_deploy.prototxt"
ageModel= path + "age_net.caffemodel"

genderProto= path + "gender_deploy.prototxt"
genderModel= path + "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# List of age 
# MODEL_MEAN_VALUES is the mean of each color channel across the ImageNet dataset
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-23)', '(25-34)', '(38-46)', '(48-55)','(58-71)', '(75-90)']
genderList = ['Male','Female']

# Open webcam
video=cv2.VideoCapture(0)
padding=20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                   :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

video.release()
cv2.destroyAllWindows()
