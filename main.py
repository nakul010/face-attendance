import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import model

def capture():
    cam = cv2.VideoCapture(0)
    folder = "dataset/image/people/"
    name = ""
    temp = False

    while cam.isOpened():
        result, frame = cam.read()

        if not result:
            print("Failed to open camera")
            break

        cv2.imshow("Capturing", frame)

        if cv2.waitKey(1)%256 == 32:
            face = frame
            temp = True
            print("Captured")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()

    try:
        if temp:
            name = input("What name should this image be given: ")
            cv2.imwrite(f"{folder}{name}.png", face)
            print(f"Save {name}.png")
    except:
        print("Someting went wrong. Please retry")

    return name

# add_image = input("Do ypu want to add new member: ").lower()
# if add_image == "y":
#     name = capture()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("dataset/video/test.mp4")

# cap.set(3, 640)
# cap.set(4, 360)

# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
# # # Calculate the desired window size based on the aspect ratio
# # desired_width = 800  # Adjust this value as per your requirements
# # desired_height = int(height * desired_width / width)
#
# # Set the window size
# cv2.resizeWindow("Video", desired_width, desired_height)

# img_back = cv2.imread("com.jpg")
# if not cap.isOpened():
#     print("Error")

file = open("model_file.p", "rb")
encoded_id = pickle.load(file)
file.close()
encoded, image_id = encoded_id


# while cap.isOpened():
while True:
    success, img = cap.read()

    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(img_small)  # faces in the current frame
    encodeCurFrame = face_recognition.face_encodings(img_small, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encoded, encodeFace)
        faceDis = face_recognition.face_distance(encoded, encodeFace)
        print(matches)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis >= .6:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bounding_box = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            img = cvzone.cornerRect(img, bounding_box, rt=0)
            print(image_id)

    # img_back[60:60+360,373:373+640] = img
    cv2.imshow("Face", img)
    # print(cvzone.FPS)
    # cv2.imshow("Background", img_back)
    # cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
