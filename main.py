import cv2

#only need face and mouth cascade to detect mask/without
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')



#font size, type, message for no face location, thickness, mask message
font = cv2.FONT_HERSHEY_SIMPLEX
top = (40,40)
thickness = 2
font_scale = 1
with_mask_font_color = (255, 255, 255)
without_mask_font_color = (0, 0, 255)
with_mask_text = "Mask"
without_mask_text = "No Mask"

# Read video
cap = cv2.VideoCapture(0)

while True:
    #basic read operation to get consistent frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    #changes frames to gray to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #there is no length faces means no face
    if(len(faces) == 0 ):
        cv2.putText(frame, "No faces found.", top, font, font_scale, with_mask_font_color, thickness, cv2.LINE_AA)

    #else there is a face

    else:
        # Draw rectangle on gace
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]


            # check for
            mouth = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # if theres a face but no mouth write text to top
        if(len(mouth) == 0):
            cv2.putText(frame, with_mask_text, top, font, font_scale, with_mask_font_color, thickness, cv2.LINE_AA)

        else:
            for (ex, ey, ew, eh) in mouth:
                if(y < ey < y + h):
                    # mouth and face are present but mouth is in face so there is mouth = person no mask
                    
                    cv2.putText(frame, without_mask_text, top, font, font_scale, without_mask_font_color, thickness, cv2.LINE_AA)

                    break


    cv2.imshow('Mask Detection', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()