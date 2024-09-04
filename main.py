import os
import pickle
import cv2
import warnings
import mediapipe as mp
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Load the model and label mapping
with open('./model2.p', 'rb') as file:
    model_dict = pickle.load(file)
model = model_dict['model']
label_mapping = model_dict['label_mapping']
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize variables
recognized_text = ""
last_sign = ""
last_sign_time = cv2.getTickCount()  # Initialize last sign time

def add_text_to_image(text, image):
    # Set the position of the text
    x, y, dy = 50, 50, 30
    max_width = image.shape[1] - 100  # Margin from left and right
    
    # Break text into lines
    lines = []
    words = text.split(' ')
    line = ''
    
    for word in words:
        # Test line width with the new word
        test_line = f"{line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        if w < max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    
    if line:
        lines.append(line)
    
    # Add lines to image
    for i, line in enumerate(lines):
        cv2.putText(image, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    # Prepare a white screen for text display
    white_screen = np.ones_like(frame) * 255

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]

            data_aux = [(x - min(x_)) for x in x_]
            data_aux += [(y - min(y_)) for y in y_]

        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = prediction[0]

            if isinstance(predicted_index, str) and predicted_index in reverse_label_mapping:
                predicted_character = predicted_index
            elif isinstance(predicted_index, int) and predicted_index in reverse_label_mapping.values():
                predicted_character = label_mapping[predicted_index]
            else:
                predicted_character = "Unknown"

            current_time = cv2.getTickCount()
            time_diff = (current_time - last_sign_time) / cv2.getTickFrequency()

            if predicted_character != "Unknown" and (predicted_character != last_sign or time_diff > 2):  # 2 seconds threshold
                # Update recognized text only if the sign has changed
                if predicted_character != last_sign:
                    recognized_text += predicted_character + ' '
                    last_sign = predicted_character
                    last_sign_time = current_time

    # Add recognized text to the white screen
    add_text_to_image(recognized_text.strip(), white_screen)

    # Display both webcam feed and text
    cv2.imshow('Webcam Feed', frame)
    cv2.imshow('Recognized Signs', white_screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
