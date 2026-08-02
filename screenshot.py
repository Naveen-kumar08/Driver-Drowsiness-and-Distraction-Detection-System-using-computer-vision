import cv2
import os
from datetime import datetime


class ScreenshotManager:

    def __init__(self):

        self.folder = "output/screenshots"

        os.makedirs(self.folder, exist_ok=True)

        self.last_save = None

        self.delay = 3      # Save only once every 3 seconds

    def save(self, frame, status):

        current_time = datetime.now()

        # Prevent saving too many screenshots
        if self.last_save is not None:

            seconds = (current_time - self.last_save).total_seconds()

            if seconds < self.delay:

                return

        self.last_save = current_time

        filename = current_time.strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        filename += "_" + status + ".jpg"

        filepath = os.path.join(
            self.folder,
            filename
        )

        cv2.imwrite(
            filepath,
            frame
        )

        print(f"[SCREENSHOT] Saved : {filepath}")