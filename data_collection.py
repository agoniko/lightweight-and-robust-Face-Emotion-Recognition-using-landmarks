import mediapipe as mp
import cv2
import numpy as np
from pynput import keyboard
from time import time
import json
import os

USER = "Nic"
landmark_path = "custom_dataset/test/landmark_dataset.json"
video_path = "custom_dataset/test/video_dataset/"


EMOTION = None
LAST_PRESSED = None
landmarks = []


key2emotion = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "surprised",
    5: "disgusted",
}

def key_press(key):
    global EMOTION, LAST_PRESSED
    try:
        k = key.char
        if k == "q":
            return False
        if k.isdigit() and int(k) in key2emotion:
            EMOTION = key2emotion[int(k)]
            LAST_PRESSED = time()
    except AttributeError:
        print("Special key {0} pressed".format(key))
 

def landmark_to_array(landmarks):
    l_tensor = np.zeros((478, 3))

    for i, landmark in enumerate(landmarks.landmark):
        l_tensor[i] = np.array([landmark.x, landmark.y, landmark.z])

    return l_tensor

def mp_to_json(landmarks, emotion):
    l_tensor = landmark_to_array(landmarks)
    return {"landmarks": l_tensor.tolist(), "emotion": emotion}


def collect_data():
    global EMOTION, LAST_PRESSED, landmarks, images
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    count = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    images = []

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                break


            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame.copy()
            results = face_mesh.process(image)
            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                # Flip the image horizontally for a selfie-view display.
                #image = cv2.resize(image, (720, 480))
                cv2.putText(
                    image,
                    f"Emotion: {EMOTION}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("MediaPipe Face Mesh", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            if LAST_PRESSED and time() - LAST_PRESSED < 1:
                landmarks.append(mp_to_json(face_landmarks, EMOTION))
                images.append(frame)
            else:
                EMOTION = None
        
    print(len(landmarks), len(images))
    assert len(images) == len(landmarks)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    with keyboard.Listener(on_press=key_press) as listener:
        collect_data()
        listener.join()

    with open(landmark_path, "r") as f:
        data = json.load(f)
        old_data = data.get(USER, [])
        print(len(old_data))
        data[USER] = old_data + landmarks
    
    print(len(data[USER]))
    
    with open(landmark_path, "w") as f:
        json.dump(data, f)
    print("Landmarks saved")

    for image, landmark in zip(images, landmarks):
        emotion = landmark["emotion"]
        os.makedirs(video_path + emotion, exist_ok=True)
        cv2.imwrite(video_path + emotion + f"/{time()}.jpg", image)
    
    print("Images saved")


    exit(0)