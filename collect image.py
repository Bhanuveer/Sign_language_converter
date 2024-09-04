import os
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'OK']
letters = ['Ok' 'Hello', 'I Love You', 'Telephone', 'Money', 'Love', 'yes','No', 'bathroom', 'Iam', 'why', 'water', 'you', 'errase']
number_of_classes = len(letters)
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for letter in letters:
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

class_counters = {}
for j, letter in enumerate(letters):
    class_dir = os.path.join(DATA_DIR, letter)
    class_counters[letter] = len(os.listdir(class_dir))

for j, letter in enumerate(letters):
    print(f'Collecting data for letter {letter}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        cv2.putText(frame, f'Ready? Press "Q" to capture letter {letter}!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    counter = class_counters[letter]
    while counter < class_counters[letter] + dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, letter, '{}.jpg'.format(counter)), frame)
        print(f"Captured frame {counter} for letter {letter}")

        counter += 1

cap.release()
cv2.destroyAllWindows()