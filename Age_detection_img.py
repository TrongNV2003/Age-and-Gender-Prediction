import cv2

# Prepare data input
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    
    # Run the network to get detections
    detections = net.forward()
    
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            startX=int(detections[0,0,i,3]*frameWidth)
            startY=int(detections[0,0,i,4]*frameHeight)
            endX=int(detections[0,0,i,5]*frameWidth)
            endY=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([startX,startY,endX,endY])
            cv2.rectangle(frameOpencvDnn, (startX,startY), (endX,endY), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

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

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746) #
ageList=['(0-3)', '(4-7)', '(8-14)', '(15-23)', '(25-34)', '(38-46)', '(48-60)','(62-71)', '(75-90)']
genderList=['Male','Female']

# Load the test image
img_path = 'C:/Users/Trongdz/Desktop/Age and Gender Detection/image/'
img=cv2.imread(img_path + 'ai-bao-nam-gioi-chi-duoc-de-toc-ngan-thu-qua-10-kieu-toc-dai-cho-nam-cuc-dep-va-thoi-thuong-202101060938532929.jpg')
padding=20

# Ground truth labels for age and gender (for demonstration purposes)
ground_truth_age = ['(25-34)', '(25-34)', '(25-34)', '(25-34)', '(15-23)', '(25-34)', '(25-34)', '(25-34)', '(25-34)','(15-23)', '(8-14)', '(25-34)']
ground_truth_gender = ['Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female']

# Perform face detection and age/gender classification
resultImg,faceBoxes=highlightFace(faceNet,img)

for faceBox in faceBoxes:
    face = img[max(0,faceBox[1]-padding):
                min(faceBox[3]+padding,img.shape[0]-1),max(0,faceBox[0]-padding)
                :min(faceBox[2]+padding, img.shape[1]-1)]

    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    
    
    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("Detecting age and gender", resultImg)
    cv2.waitKey(0)

cv2.destroyAllWindows()
