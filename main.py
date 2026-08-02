import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime


# ==========================================
# Import Project Modules
# ==========================================

from eye_detection import EyeDetection
from yawn_detection import YawnDetection
from headposition import HeadPosition
from phone_detection import PhoneDetection
from logger import DriverLogger
from screenshot import ScreenshotManager
from buzzer import Buzzer
# ==========================================
# MediaPipe Face Mesh
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# Create Objects
# ==========================================

eye = EyeDetection()

yawn = YawnDetection()

head = HeadPosition()

phone = PhoneDetection()

logger = DriverLogger()

screenshot = ScreenshotManager()
buzzer = Buzzer()

# ==========================================
# Open Camera
# ==========================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():

    print("Camera Not Found")

    exit()

print("=" * 50)
print(" AI DRIVER MONITORING SYSTEM STARTED ")
print("=" * 50)

# ==========================================
# Colors
# ==========================================

GREEN = (0,255,0)
RED = (0,0,255)
YELLOW = (0,255,255)
WHITE = (255,255,255)
CYAN = (255,255,0)
BLUE = (255,0,0)
GRAY = (60,60,60)
BLACK = (35,35,35)

# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror Image

    frame = cv2.flip(frame,1)

    # RGB Conversion

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Face Mesh

    results = face_mesh.process(rgb)

    # =====================================
    # Dashboard
    # =====================================

    dashboard = np.full(
        (480,360,3),
        35,
        dtype=np.uint8
    )

    cv2.rectangle(
        dashboard,
        (0,0),
        (360,80),
        (45,45,45),
        -1
    )

    cv2.putText(
        dashboard,
        "AI DRIVER",
        (60,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        CYAN,
        2
    )

    cv2.putText(
        dashboard,
        "MONITORING SYSTEM",
        (35,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2
    )

    cv2.line(
        dashboard,
        (15,90),
        (345,90),
        WHITE,
        1
    )

    # =====================================
    # Default Values
    # =====================================

    ear = 0.0

    mar = 0.0

    head_position = "UNKNOWN"

    phone_detected = False

    phone_usage = False

    phone_info = {

        "label":"NO PHONE",

        "phone_side":"NONE"

    }

    drowsy = False

    yawning = False

    status = "ALERT"

    status_color = GREEN

    # =====================================
    # Face Detected
    # =====================================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        # =====================================
        # Eye Detection (EAR)
        # =====================================

        ear = eye.detect(
            frame,
            face_landmarks
        )

        drowsy = eye.drowsiness(
            frame,
            ear
        )

        # =====================================
        # Yawn Detection (MAR)
        # =====================================

        mar = yawn.detect(
            frame,
            face_landmarks
        )

        yawning = yawn.yawn_status(
            frame,
            mar
        )

        # =====================================
        # Head Position
        # =====================================

        head_position = head.detect(
            frame,
            face_landmarks
        )

        # =====================================
        # Phone Detection
        # =====================================

        phone_detected, phone_usage, phone_info = phone.detect(
            frame,
            face_landmarks
        )

        # Safety check
        if isinstance(phone_info, str):

            phone_info = {
                "label": phone_info,
                "phone_side": "NONE"
            }

        # =====================================
        # Driver Status
        # =====================================

        if drowsy:

            status = "DROWSY"
            status_color = RED

        elif yawning:

            status = "YAWNING"
            status_color = YELLOW

        elif phone_usage:

            status = "DISTRACTED"
            status_color = RED

        elif head_position != "LOOKING FORWARD":

            status = "DISTRACTED"
            status_color = RED

        else:

            status = "ALERT"
            status_color = GREEN

        # =====================================
        # Camera Border
        # =====================================

        cv2.rectangle(
            frame,
            (5,5),
            (635,475),
            status_color,
            3
        )

        # =====================================
        # Time
        # =====================================

        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        cv2.putText(
            frame,
            current_time,
            (10,470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )
        # =====================================
        # Dashboard Information
        # =====================================

        cv2.putText(
            dashboard,
            f"EAR  : {ear:.2f}",
            (30,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )

        cv2.putText(
            dashboard,
            f"MAR  : {mar:.2f}",
            (30,170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )

        cv2.putText(
            dashboard,
            f"HEAD : {head_position}",
            (30,210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2
        )

        # =====================================
        # Phone Information
        # =====================================

        phone_text = phone_info.get("label", "NO PHONE")

        if phone_usage:

            phone_color = RED

        elif phone_detected:

            phone_color = YELLOW

        else:

            phone_color = GREEN

        cv2.putText(
            dashboard,
            phone_text,
            (30,250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            phone_color,
            2
        )

        # =====================================
        # Driver Status Box
        # =====================================

        cv2.rectangle(
            dashboard,
            (20,290),
            (340,420),
            (60,60,60),
            -1
        )

        cv2.rectangle(
            dashboard,
            (20,290),
            (340,420),
            status_color,
            2
        )

        cv2.putText(
            dashboard,
            "DRIVER STATUS",
            (70,320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            WHITE,
            2
        )

        cv2.putText(
            dashboard,
            status,
            (80,370),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            3
        )

        # =====================================
        # Warning Message
        # =====================================

        if status == "ALERT":

            warning = "Driver is Safe"

        elif status == "DROWSY":

            warning = "Wake Up!"

        elif status == "YAWNING":

            warning = "Take a Break"

        else:

            warning = "Pay Attention"
        # =====================================
# Laptop Buzzer Control
# =====================================

        if status == "ALERT":

            buzzer.off()

        else:

            buzzer.on()   

        cv2.putText(
            dashboard,
            warning,
            (40,405),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2
        )

        # =====================================
        # Save Screenshot
        # =====================================

        if status != "ALERT":

            screenshot.save(
                frame,
                status
            )

        # =====================================
        # Save CSV Log
        # =====================================

        logger.save(
            ear,
            mar,
            head_position,
            phone_text,
            status
        )

    # =====================================
    # No Face Detected
    # =====================================

    else:

        status = "NO DRIVER"

        status_color = RED

        cv2.putText(
            dashboard,
            "NO DRIVER DETECTED",
            (30,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            RED,
            2
        )

        cv2.putText(
            dashboard,
            "Please Sit Properly",
            (45,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )
    # =====================================
    # Footer
    # =====================================

    cv2.line(
        dashboard,
        (20,440),
        (340,440),
        (120,120,120),
        1
    )

    cv2.putText(
        dashboard,
        "Press Q to Exit",
        (30,465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    # =====================================
    # Camera Title
    # =====================================

    cv2.putText(
        frame,
        "LIVE CAMERA",
        (10,25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        CYAN,
        2
    )

    # =====================================
    # Dashboard Title
    # =====================================

    cv2.putText(
        dashboard,
        "SYSTEM DASHBOARD",
        (140,465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        CYAN,
        1
    )

    # =====================================
    # Combine Camera + Dashboard
    # =====================================

    output = np.hstack((frame, dashboard))

    # =====================================
    # Display Window
    # =====================================

    cv2.imshow(
        "AI Driver Monitoring System",
        output
    )

    # =====================================
    # Exit Key
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# =====================================
# Release Resources
# =====================================

buzzer.off()

camera.release()

cv2.destroyAllWindows()                       