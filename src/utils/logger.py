import os
from datetime import datetime


class Logger:
    """
    Simple logger for training
    """

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

    def log(self, message):
        print(message)

        with open(self.log_file, "a") as f:
            f.write(message + "\n")