import cv2
from scipy.spatial import distance


class EyeDetection:

    def __init__(self):

        # Left Eye Landmarks
        self.LEFT_EYE = [
            33, 160, 158,
            133, 153, 144
        ]

        # Right Eye Landmarks
        self.RIGHT_EYE = [
            362, 385, 387,
            263, 373, 380
        ]

        # EAR Threshold
        self.EAR_THRESHOLD = 0.20

        # Number of consecutive frames
        self.FRAME_THRESHOLD = 20

        self.counter = 0

    # ==================================
    # Calculate Eye Aspect Ratio
    # ==================================

    def calculate_ear(self, eye):

        A = distance.euclidean(
            eye[1],
            eye[5]
        )

        B = distance.euclidean(
            eye[2],
            eye[4]
        )

        C = distance.euclidean(
            eye[0],
            eye[3]
        )

        ear = (A + B) / (2.0 * C)

        return ear

    # ==================================
    # Detect Eyes
    # ==================================

    def detect(self, frame, face_landmarks):

        h, w, _ = frame.shape

        left_eye = []

        right_eye = []

        # --------------------------
        # Left Eye
        # --------------------------

        for idx in self.LEFT_EYE:

            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)

            left_eye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        # --------------------------
        # Right Eye
        # --------------------------

        for idx in self.RIGHT_EYE:

            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)

            right_eye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        leftEAR = self.calculate_ear(left_eye)

        rightEAR = self.calculate_ear(right_eye)

        ear = (leftEAR + rightEAR) / 2

        return ear

    # ==================================
    # Drowsiness Detection
    # ==================================

    def drowsiness(self, frame, ear):

        if ear < self.EAR_THRESHOLD:

            self.counter += 1

        else:

            self.counter = 0

        if self.counter >= self.FRAME_THRESHOLD:

            return True

        return False