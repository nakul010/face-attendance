import cv2
import face_recognition
import os
import pickle

image_folder = "dataset/image/people/"
image_path = os.listdir(image_folder)
image_list = []
image_id = []
for images in image_path:
    # image_list.append(images)
    image_list.append(cv2.imread(os.path.join(image_folder, images)))
    image_id.append(os.path.splitext(images)[0])


# print(image_list)
# print(image_id)

def model(images_list):
    encode_list = []
    for image in image_list:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode_list.append(face_recognition.face_encodings(img)[0])
    return encode_list


encoded = model(image_list)
encoded_id = [encoded, image_id]
# print(encoded_id)

file = open("model_file.p", "wb")
pickle.dump(encoded_id, file)
file.close()
# print("File saved")