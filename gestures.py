
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=max_num_hands,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)

    def find_hands(self, frame, draw=True):
        """
        Runs MediaPipe hand detection on a BGR frame.
        Returns list of hands: each is a dict with 'landmarks' (list of (x,y) in pixels) and 'handedness' ('Left'/'Right')
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        hands_out = []
        if res.multi_hand_landmarks:
            for i, handLms in enumerate(res.multi_hand_landmarks):
                lms = []
                for lm in handLms.landmark:
                    lms.append((int(lm.x * w), int(lm.y * h), lm.z))
                # handedness
                handed = 'Unknown'
                if res.multi_handedness and i < len(res.multi_handedness):
                    handed = res.multi_handedness[i].classification[0].label
                hands_out.append({'landmarks': lms, 'handedness': handed, 'mp_landmarks': handLms})
        return hands_out

    @staticmethod
    def landmark_distance(lm1, lm2):
        x1, y1, *_ = lm1
        x2, y2, *_ = lm2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def is_pinch(self, hand, threshold_px=40):
        # tip of thumb: 4, tip of index: 8
        lms = hand['landmarks']
        d = self.landmark_distance(lms[4], lms[8])
        return d < threshold_px, d

    def fingers_up_count(self, hand):
        # naive fingers up detection using landmarks: compare tip y to pip y
        lms = hand['landmarks']
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        count = 0
        try:
            for tip, pip in zip(tips, pips):
                if lms[tip][1] < lms[pip][1]:
                    count += 1
        except Exception:
            return 0
        return count

    def hands_center_distance(self, hand1, hand2):
        # compute average of wrist points (0) or average of several landmarks
        c1 = np.mean(np.array([(x, y) for x, y, _ in hand1['landmarks']]), axis=0)
        c2 = np.mean(np.array([(x, y) for x, y, _ in hand2['landmarks']]), axis=0)
        return np.linalg.norm(c1 - c2)

    def classify(self, hands):
        """
        Returns a dict with simplified gesture info:
         - 'pinch': (True/False, distance_px) for each hand index
         - 'open_count': fingers up count
         - 'two_hands_distance': if two hands present
        """
        out = {'hands': [], 'two_hands_distance': None}
        for h in hands:
            is_p, d = self.is_pinch(h)
            up = self.fingers_up_count(h)
            out['hands'].append({'handedness': h.get('handedness', 'Unknown'), 'is_pinch': is_p, 'pinch_dist': d, 'fingers_up': up})
        if len(hands) >= 2:
            out['two_hands_distance'] = self.hands_center_distance(hands[0], hands[1])
        return out
