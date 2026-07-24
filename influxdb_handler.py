from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

class InfluxDBHandler:
    
    def __init__(self):
        self.url    = "http://10.0.30.7:8086"
        self.token  = "gCIyOVHn5r9NXM1vjKgA3c287agvLRO33knaGJGO35wVYGbpcL3wuJ_sIkYz7Q-EdjesvxWt3_uUAaD_IxiIng=="
        self.org    = "EmoSys"
        self.bucket = "emotionDB"

        self.client = InfluxDBClient(
            url    = self.url,
            token  = self.token,
            org = self.org
        )

        self.write_api  = self.client.write_api(
            write_options = SYNCHRONOUS
        )

        self.last_saved = 0
        self.interval   = 5 #seconds

    def write_prediction(self, emotion, confidence):

        current_time = time.time()

        if current_time - self.last_saved < self.interval:
            return

        point = (
            Point("FER Prediction")
            .tag("device", "pi1")
            .field("emotion", emotion)
            .field("confidence", float(confidence))
        )

        # Added a try-except block just in case InfluxDB is offline so the camera doesn't crash
        try:
            self.write_api.write(
                bucket = self.bucket,
                org    = self.org,
                record = point
            )
            self.last_saved = current_time
            print(f"InfluxDB saved: {emotion} ({confidence:.2f})")
        except Exception as e:
            print(f"InfluxDB save failed: {e}")

    def close(self):
        self.write_api.close()
        self.client.close()
