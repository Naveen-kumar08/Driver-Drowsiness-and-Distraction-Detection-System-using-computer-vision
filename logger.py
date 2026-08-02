import csv
import os
from datetime import datetime


class DriverLogger:

    def __init__(self):

        # Create output folder
        self.folder = "output/logs"
        os.makedirs(self.folder, exist_ok=True)

        # CSV file
        self.filename = os.path.join(
            self.folder,
            "driver_log.csv"
        )

        # Create file if it doesn't exist
        if not os.path.exists(self.filename):

            with open(
                self.filename,
                mode="w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Date",
                    "Time",
                    "EAR",
                    "MAR",
                    "Head Position",
                    "Phone",
                    "Driver Status"
                ])

    # ==========================================
    # Save Data
    # ==========================================

    def save(
        self,
        ear,
        mar,
        head_position,
        phone,
        status
    ):

        now = datetime.now()

        date = now.strftime("%d-%m-%Y")

        time = now.strftime("%H:%M:%S")

        with open(
            self.filename,
            mode="a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                date,

                time,

                f"{ear:.3f}",

                f"{mar:.3f}",

                head_position,

                phone,

                status

            ])