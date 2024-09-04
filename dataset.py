import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
                y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
                
                data_aux = [(x - min(x_)) for x in x_]
                data_aux += [(y - min(y_)) for y in y_]
                
            if len(data_aux) != 42:
                continue

            data.append(data_aux)
            labels.append(dir_)

data = np.asarray(data)
labels = np.asarray(labels)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created with {len(data)} samples")