import math
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

keyboard_layout = [
    [{"key": "`", "width": 1}, {"key": "1", "width": 1}, {"key": "2", "width": 1}, {"key": "3", "width": 1},
     {"key": "4", "width": 1}, {"key": "5", "width": 1}, {"key": "6", "width": 1}, {"key": "7", "width": 1},
     {"key": "8", "width": 1}, {"key": "9", "width": 1}, {"key": "0", "width": 1}, {"key": "-", "width": 1},
     {"key": "=", "width": 1}, {"key": "Back", "width": 2}],
    [{"key": "Tab", "width": 1.5}, {"key": "Q", "width": 1}, {"key": "W", "width": 1}, {"key": "E", "width": 1},
     {"key": "R", "width": 1}, {"key": "T", "width": 1}, {"key": "Y", "width": 1}, {"key": "U", "width": 1},
     {"key": "I", "width": 1}, {"key": "O", "width": 1}, {"key": "P", "width": 1}, {"key": "[", "width": 1},
     {"key": "]", "width": 1}, {"key": "\\", "width": 1}],
    [{"key": "Caps", "width": 1.75}, {"key": "A", "width": 1}, {"key": "S", "width": 1}, {"key": "D", "width": 1},
     {"key": "F", "width": 1}, {"key": "G", "width": 1}, {"key": "H", "width": 1}, {"key": "J", "width": 1},
     {"key": "K", "width": 1}, {"key": "L", "width": 1}, {"key": ";", "width": 1}, {"key": "'", "width": 1},
     {"key": "Enter", "width": 2.25}],
    [{"key": "Shift", "width": 2.25}, {"key": "Z", "width": 1}, {"key": "X", "width": 1}, {"key": "C", "width": 1},
     {"key": "V", "width": 1}, {"key": "B", "width": 1}, {"key": "N", "width": 1}, {"key": "M", "width": 1},
     {"key": ",", "width": 1}, {"key": ".", "width": 1}, {"key": "/", "width": 1}, {"key": "Shift", "width": 2.75}],
    [{"key": "Ctrl", "width": 1.25}, {"key": "Alt", "width": 1.25}, {"key": "Space", "width": 6.25},
     {"key": "Alt", "width": 1.25}, {"key": "Ctrl", "width": 1.25}]
]

smallk = ["Ctrl", "Alt","Caps","Tab","Back"]

key_size = 30
key_margin = 10
start_x, start_y = 30, 50 

width, height = 600, 400
origin = (2, 2)

def draw_keyboard(image):
    for row_idx, row in enumerate(keyboard_layout):
        x_offset = start_x
        for key_info in row:
            key = key_info["key"]
            key_width = int(key_info["width"] * key_size)

            y = start_y + row_idx * (key_size + key_margin)
            cv2.rectangle(image, (x_offset, y), (x_offset + key_width, y + key_size), (255, 100, 0), 2)

            size = 0.5 if key in smallk  else 0.7
            cv2.putText(image, key, (x_offset + 7, y + 22), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2)

            x_offset += key_width + key_margin

def detect_key(landmark, image):
    for row_idx, row in enumerate(keyboard_layout):
        x_offset = start_x
        for key_info in row:
            key = key_info["key"]
            key_width = int(key_info["width"] * key_size)

            y = start_y + row_idx * (key_size + key_margin)
            if x_offset < landmark.x * image.shape[1] < x_offset + key_width and y < landmark.y * image.shape[0] < y + key_size:
                return key

            x_offset += key_width + key_margin
    return None


# Open the camera
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    draw_keyboard(frame)
    cv2.line(frame, (5, 5), (5, height + 50), (255, 255, 255), 2)
    cv2.line(frame, (5,height + 50), (width + 20, height + 50), (255, 255, 255), 2) 

    pressed_key_info = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]

            thumb_mcp = hand_landmarks.landmark[2]
            index_finger_dip = hand_landmarks.landmark[7]

            main = hand_landmarks.landmark[0]
            main2 = hand_landmarks.landmark[12]
            main3 = hand_landmarks.landmark[9]

            main_coord = (int(main.x * frame.shape[1]), int(main.y * frame.shape[0]))
            main2_coord = (int(main2.x * frame.shape[1]), int(main2.y * frame.shape[0]))
            main3_cord = (int(main3.x * frame.shape[1]), int(main3.y * frame.shape[0]))


            distancemain = math.sqrt(
                (main.x - main2.x) ** 2 +
                (main.y - main2.y) ** 2 +
                (main.z - main2.z) ** 2
            )

            cv2.circle(frame, main3_cord, int(round(distancemain, 2) * 300), (255, 255, 255), 1)

            #0.6-0.2



            thumb_tip_coord = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_tip_coord = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
            thumb_mcp_coord = (int(thumb_mcp.x * frame.shape[1]), int(thumb_mcp.y * frame.shape[0]))
            index_dip_coord = (int(index_finger_dip.x * frame.shape[1]), int(index_finger_dip.y * frame.shape[0]))

            distance = math.sqrt(
                (thumb_tip.x - index_finger_tip.x) ** 2 +
                (thumb_tip.y - index_finger_tip.y) ** 2 +
                (thumb_tip.z - index_finger_tip.z) ** 2
            )

            cv2.line(frame, thumb_mcp_coord, thumb_tip_coord, (255, 255, 255), 2)  # Thumb
            cv2.line(frame, index_dip_coord, index_tip_coord, (255, 255, 255), 2)  # Index finger

            max_distance = 0.5
            intensity = max(0, 255 - int((distance / max_distance) ** 3 * 255))

            intensity = max(0, min(255, intensity))


            cv2.line(frame, thumb_tip_coord, index_tip_coord, (intensity, 0, 0), 2)

            mainx = round(index_finger_tip.x * frame.shape[1])
            mainy = round(index_finger_tip.y * frame.shape[0])

            cv2.line(frame, (mainx, mainy), (5, mainy), (255, 255, 255), 1)
            cv2.line(frame, (mainx, mainy), (mainx, height + 50), (255, 255, 255), 1)


            landmarks_to_highlight = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]

            for landmark_id in landmarks_to_highlight:
                landmark = hand_landmarks.landmark[landmark_id]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (x,y), 30, (255, 255, 255), 1)

            index_finger_tip = hand_landmarks.landmark[8]
            pressed_key = detect_key(index_finger_tip, frame)
            if pressed_key and distance < 0.15:
                print(f"Key Pressed: {pressed_key}")
                for row_idx, row in enumerate(keyboard_layout):
                    x_offset = start_x
                    for key_info in row:
                        key = key_info["key"]
                        key_width = int(key_info["width"] * key_size)
                        y = start_y + row_idx * (key_size + key_margin)

                        if key == pressed_key:
                            pressed_key_info = (x_offset, y, key_width, key_size)
                            break  # Stop the loop once the pressed key is found
                        
                        x_offset += key_width + key_margin

    draw_keyboard(frame)

    if pressed_key_info:
        x, y, key_width, key_size = pressed_key_info
        cv2.rectangle(frame, (x, y), (x + key_width, y + key_size), (255, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x + key_width, y + key_size), (255, 255, 255), 2)  # White border

    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key press
        break

cap.release()
cv2.destroyAllWindows()
