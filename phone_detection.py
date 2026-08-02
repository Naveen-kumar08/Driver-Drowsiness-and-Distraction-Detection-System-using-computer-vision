import cv2
import math
import time
import mediapipe as mp
from ultralytics import YOLO


class PhoneDetection:

    def __init__(self):

        # ==============================
        # Load YOLO Model
        # ==============================

        self.model = YOLO("models/yolov8n.pt")

        # COCO class id for cell phone
        self.PHONE_CLASS = 67

        self.CONFIDENCE = 0.45

        # ==============================
        # MediaPipe Hands
        # ==============================

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=2,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )

        # ==============================
        # Face Landmark IDs
        # ==============================

        self.NOSE = 1

        self.LEFT_EAR = 234

        self.RIGHT_EAR = 454

        self.LEFT_EYE = 33

        self.RIGHT_EYE = 263

        self.MOUTH = 13

        # ==============================
        # Timer
        # ==============================

        self.phone_start = None

        self.TIME_THRESHOLD = 2.0

        self.reset()

    # ===========================================
    # Reset Variables
    # ===========================================

    def reset(self):

        self.phone_detected = False

        self.phone_usage = False

        self.phone_call = False

        self.texting = False

        self.phone_hand = False

        self.phone_side = "NONE"

    # ===========================================
    # Distance Function
    # ===========================================

    def distance(self, p1, p2):

        return math.sqrt(

            (p1[0]-p2[0])**2 +

            (p1[1]-p2[1])**2

        )

    # ===========================================
    # Convert Landmark to Pixel
    # ===========================================

    def point(self, face_landmarks, index, w, h):

        x = int(

            face_landmarks.landmark[index].x * w

        )

        y = int(

            face_landmarks.landmark[index].y * h

        )

        return (x, y)

    # ===========================================
    # Main Detection Function
    # ===========================================

    def detect(self, frame, face_landmarks):

        self.reset()

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )
                # ===========================================
        # Face Landmark Points
        # ===========================================

        nose = self.point(face_landmarks, self.NOSE, w, h)

        left_ear = self.point(face_landmarks, self.LEFT_EAR, w, h)

        right_ear = self.point(face_landmarks, self.RIGHT_EAR, w, h)

        left_eye = self.point(face_landmarks, self.LEFT_EYE, w, h)

        right_eye = self.point(face_landmarks, self.RIGHT_EYE, w, h)

        mouth = self.point(face_landmarks, self.MOUTH, w, h)

        # ===========================================
        # YOLO Detection
        # ===========================================

        results = self.model(frame, verbose=False)

        phone_center = None

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                if cls != self.PHONE_CLASS:

                    continue

                if conf < self.CONFIDENCE:

                    continue

                self.phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                phone_center = (

                    (x1 + x2) // 2,

                    (y1 + y2) // 2

                )

                # Draw Phone Box

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 255),

                    2

                )

                cv2.putText(

                    frame,

                    f"PHONE {conf:.2f}",

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0,255,255),

                    2

                )

                cv2.circle(

                    frame,

                    phone_center,

                    5,

                    (0,0,255),

                    -1

                )

        # ===========================================
        # No Phone Found
        # ===========================================

        if phone_center is None:

            self.phone_start = None

            return (

                False,

                False,

                {

                    "label": "NO PHONE",

                    "phone_call": False,

                    "texting": False,

                    "phone_hand": False,

                    "phone_side": "NONE"

                }

            )

        # ===========================================
        # Distance Calculations
        # ===========================================

        d_left_ear = self.distance(phone_center, left_ear)

        d_right_ear = self.distance(phone_center, right_ear)

        d_nose = self.distance(phone_center, nose)

        d_mouth = self.distance(phone_center, mouth)

        d_left_eye = self.distance(phone_center, left_eye)

        d_right_eye = self.distance(phone_center, right_eye)

        # ===========================================
        # Phone Side
        # ===========================================

        if phone_center[0] < nose[0]:

            self.phone_side = "LEFT"

        else:

            self.phone_side = "RIGHT"

        # ===========================================
        # Phone Call Detection
        # ===========================================

        EAR_DISTANCE = 120

        if d_left_ear < EAR_DISTANCE:

            self.phone_call = True

            self.phone_side = "LEFT"

        elif d_right_ear < EAR_DISTANCE:

            self.phone_call = True

            self.phone_side = "RIGHT"

        # ===========================================
        # Phone Near Face
        # ===========================================

        if (

            d_nose < 140 or

            d_mouth < 120 or

            d_left_eye < 120 or

            d_right_eye < 120

        ):

            self.phone_usage = True
                    # ===========================================
        # MediaPipe Hands Detection
        # ===========================================

        hand_results = self.hands.process(rgb)

        if hand_results.multi_hand_landmarks:

            for hand_landmarks in hand_results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:

                    hx = int(landmark.x * w)

                    hy = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (hx, hy),
                        2,
                        (255, 0, 255),
                        -1
                    )

                    # Phone close to hand?
                    if self.distance(phone_center, (hx, hy)) < 80:

                        self.phone_hand = True

        # ===========================================
        # Texting Detection
        # ===========================================

        if (

            phone_center[1] > mouth[1] + 40 and

            self.phone_hand

        ):

            self.texting = True

        # ===========================================
        # Usage Decision
        # ===========================================

        if (

            self.phone_call or

            self.texting or

            self.phone_hand or

            self.phone_usage

        ):

            if self.phone_start is None:

                self.phone_start = time.time()

            elapsed = time.time() - self.phone_start

            if elapsed >= self.TIME_THRESHOLD:

                self.phone_usage = True

        else:

            self.phone_start = None

            self.phone_usage = False

        # ===========================================
        # Display Label
        # ===========================================

        label = "PHONE DETECTED"

        color = (0,255,255)

        if self.phone_call:

            label = f"PHONE CALL ({self.phone_side})"

            color = (0,0,255)

        elif self.texting:

            label = "TEXTING"

            color = (255,0,0)

        elif self.phone_hand:

            label = "PHONE IN HAND"

            color = (0,165,255)

        elif self.phone_usage:

            label = "PHONE NEAR FACE"

            color = (0,0,255)

        cv2.putText(

            frame,

            label,

            (20, frame.shape[0]-20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )

        # ===========================================
        # Return Dictionary
        # ===========================================

        phone_info = {

            "label": label,

            "phone_call": self.phone_call,

            "texting": self.texting,

            "phone_hand": self.phone_hand,

            "phone_side": self.phone_side

        }

        return (

            self.phone_detected,

            self.phone_usage,

            phone_info

        )