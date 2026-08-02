import cv2

# MediaPipe Iris Landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye Corner Landmarks
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263


class GazeTracking:

    def __init__(self):

        self.threshold_left = 0.40
        self.threshold_right = 0.60

    def detect(self, frame, face_landmarks):

        h, w, _ = frame.shape

        # -----------------------------
        # LEFT IRIS CENTER
        # -----------------------------

        iris_x = 0
        iris_y = 0

        for idx in LEFT_IRIS:

            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)

            iris_x += x
            iris_y += y

            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

        iris_x //= 4
        iris_y //= 4

        cv2.circle(frame, (iris_x, iris_y), 4, (0, 255, 255), -1)

        # -----------------------------
        # Eye Corners
        # -----------------------------

        left_corner = (
            int(face_landmarks.landmark[LEFT_EYE_LEFT].x * w),
            int(face_landmarks.landmark[LEFT_EYE_LEFT].y * h)
        )

        right_corner = (
            int(face_landmarks.landmark[LEFT_EYE_RIGHT].x * w),
            int(face_landmarks.landmark[LEFT_EYE_RIGHT].y * h)
        )

        cv2.circle(frame, left_corner, 3, (0,255,0), -1)
        cv2.circle(frame, right_corner, 3, (0,255,0), -1)

        eye_width = right_corner[0] - left_corner[0]

        if eye_width <= 0:
            return "UNKNOWN"

        ratio = (iris_x - left_corner[0]) / eye_width

        # -----------------------------
        # Gaze Direction
        # -----------------------------

        if ratio < self.threshold_left:

            gaze = "LOOKING LEFT"
            color = (0,0,255)

        elif ratio > self.threshold_right:

            gaze = "LOOKING RIGHT"
            color = (0,0,255)

        else:

            gaze = "LOOKING FORWARD"
            color = (0,255,0)

        cv2.putText(
            frame,
            gaze,
            (20,370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        return gaze