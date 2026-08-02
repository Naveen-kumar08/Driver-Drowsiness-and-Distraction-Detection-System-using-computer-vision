import cv2


class HeadPosition:

    def __init__(self):

        # MediaPipe Face Landmarks
        self.NOSE = 1
        self.CHIN = 152
        self.LEFT_FACE = 234
        self.RIGHT_FACE = 454
        self.FOREHEAD = 10

    # =====================================
    # Head Position Detection
    # =====================================

    def detect(self, frame, face_landmarks):

        h, w, _ = frame.shape

        # Nose
        nose = (
            int(face_landmarks.landmark[self.NOSE].x * w),
            int(face_landmarks.landmark[self.NOSE].y * h)
        )

        # Chin
        chin = (
            int(face_landmarks.landmark[self.CHIN].x * w),
            int(face_landmarks.landmark[self.CHIN].y * h)
        )

        # Left Face
        left_face = (
            int(face_landmarks.landmark[self.LEFT_FACE].x * w),
            int(face_landmarks.landmark[self.LEFT_FACE].y * h)
        )

        # Right Face
        right_face = (
            int(face_landmarks.landmark[self.RIGHT_FACE].x * w),
            int(face_landmarks.landmark[self.RIGHT_FACE].y * h)
        )

        # Forehead
        forehead = (
            int(face_landmarks.landmark[self.FOREHEAD].x * w),
            int(face_landmarks.landmark[self.FOREHEAD].y * h)
        )

        # -----------------------------------
        # Draw Landmarks
        # -----------------------------------

        cv2.circle(frame, nose, 4, (0,255,255), -1)
        cv2.circle(frame, chin, 4, (255,0,0), -1)
        cv2.circle(frame, left_face, 4, (255,0,0), -1)
        cv2.circle(frame, right_face, 4, (255,0,0), -1)
        cv2.circle(frame, forehead, 4, (0,255,0), -1)

        # -----------------------------------
        # Calculate Face Width
        # -----------------------------------

        left_x = left_face[0]
        right_x = right_face[0]

        face_width = right_x - left_x

        if face_width <= 0:
            return "LOOKING FORWARD"

        # -----------------------------------
        # Horizontal Ratio
        # -----------------------------------

        ratio_x = (nose[0] - left_x) / face_width

        # -----------------------------------
        # Vertical Ratio
        # -----------------------------------

        top_y = forehead[1]
        bottom_y = chin[1]

        face_height = bottom_y - top_y

        if face_height <= 0:
            return "LOOKING FORWARD"

        ratio_y = (nose[1] - top_y) / face_height

        # -----------------------------------
        # Head Direction
        # -----------------------------------

        if ratio_x < 0.42:
            return "LOOKING LEFT"

        elif ratio_x > 0.58:
            return "LOOKING RIGHT"

        elif ratio_y < 0.42:
            return "LOOKING UP"

        elif ratio_y > 0.67:
            return "LOOKING DOWN"

        else:
            return "LOOKING FORWARD"