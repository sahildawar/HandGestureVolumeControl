import cv2
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, c = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                x1, y1 = lm_list[4]
                x2, y2 = lm_list[8]
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(img, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [20, 180], [min_vol, max_vol])
                vol_bar = np.interp(length, [20, 180], [400, 150])
                vol_per = np.interp(length, [20, 180], [0, 100])

                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 2)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
                cv2.putText(img, f'{int(vol_per)} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
