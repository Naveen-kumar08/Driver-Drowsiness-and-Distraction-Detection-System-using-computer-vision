import cv2
from scipy.spatial import distance


class YawnDetection:

    def __init__(self):

        # Mouth Landmarks
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_MOUTH = 78
        self.RIGHT_MOUTH = 308

        # MAR Threshold
        self.MAR_THRESHOLD = 0.45

    # ==================================
    # Calculate Mouth Aspect Ratio
    # ==================================

    def calculate_mar(self, upper, lower, left, right):

        vertical = distance.euclidean(
            upper,
            lower
        )

        horizontal = distance.euclidean(
            left,
            right
        )

        if horizontal == 0:
            return 0.0

        mar = vertical / horizontal

        return mar

    # ==================================
    # Detect Mouth
    # ==================================

    def detect(self, frame, face_landmarks):

        h, w, _ = frame.shape

        upper = (
            int(face_landmarks.landmark[self.UPPER_LIP].x * w),
            int(face_landmarks.landmark[self.UPPER_LIP].y * h)
        )

        lower = (
            int(face_landmarks.landmark[self.LOWER_LIP].x * w),
            int(face_landmarks.landmark[self.LOWER_LIP].y * h)
        )

        left = (
            int(face_landmarks.landmark[self.LEFT_MOUTH].x * w),
            int(face_landmarks.landmark[self.LEFT_MOUTH].y * h)
        )

        right = (
            int(face_landmarks.landmark[self.RIGHT_MOUTH].x * w),
            int(face_landmarks.landmark[self.RIGHT_MOUTH].y * h)
        )

        # Draw Mouth Landmarks

        cv2.circle(frame, upper, 3, (255, 0, 0), -1)
        cv2.circle(frame, lower, 3, (255, 0, 0), -1)
        cv2.circle(frame, left, 3, (255, 0, 0), -1)
        cv2.circle(frame, right, 3, (255, 0, 0), -1)

        mar = self.calculate_mar(
            upper,
            lower,
            left,
            right
        )

        return mar

    # ==================================
    # Yawning Detection
    # ==================================

    def yawn_status(self, frame, mar):

        if mar > self.MAR_THRESHOLD:

            return True

        return False