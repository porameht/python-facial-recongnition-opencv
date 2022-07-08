
import face_recognition as face
import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

tung_image = face.load_image_file('cat.jpg')
frank_image = face.load_image_file('frank.jpg')

tung_face_encoding = face.face_encodings(tung_image)[0]
frank_face_encoding = face.face_encodings(frank_image)[0]

face_location = []
face_encoding = []
face_names = []
face_percent = []

#frame per sec
process_this_frame = True

known_face_encodings = [tung_face_encoding,frank_face_encoding]
known_face_names = ["TUNG CAT","FRANK"]

while True:
    ret, frame = video_capture.read()
    if ret:

        small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
        #convert color bgr=>opencv to rgb for calculate
        rgb_small_frame = small_frame[:,:,::-1]

        face_names = []
        face_percent = []

        if process_this_frame:
        
            face_location = face.face_locations(rgb_small_frame, model="cnn")
            face_encoding = face.face_encodings(rgb_small_frame, face_location)

            for face_encoding in face_encoding:
                face_distances = face.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_distances)
                face_percent_value = 1-face_distances[best]

                if face_percent_value >= 0.5:
                    name = known_face_names[best]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                else:
                    name = "UNKNOWN"
                    face_percent.append(0)
                face_names.append(name)    

        for (top,right,bottom,left), name, percent in zip(face_location, face_names, face_percent):
            top*=2
            right*=2
            bottom*=2
            left*=2

            if name == "UNKNOW":
                color = [45,2,210]
            else:
                color = [255,100,50]

            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame,(left-1, top -30), (right+1, top), color, cv2.FILLED)
            cv2.rectangle(frame,(left-1, bottom), (right+1, bottom+30), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.6, (155,255,255),1)
            cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (155,255,255),1)


        process_this_frame = not process_this_frame

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break

video_capture.release()
cv2.destroyAllWindows()