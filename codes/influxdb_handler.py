from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

class InfluxDBHandler:
    
    def __init__(self):
        self.url    = "http://10.0.20.159:8086"
        self.token  = "P94COCY0PTnw_oA0Dvtlv9y2ZWirIteioLJpmlkOm_PXPcX-LDO3V-Axe6mJGqIWy47uX06lP8fJKNEPOqu4cA=="
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

        # Use a dictionary to track the last saved time for EACH face ID independently
        self.last_saved_per_face = {}
        self.interval = 5 # seconds

    def write_prediction(self, face_id, emotion, confidence, posture_score, posture, gesture):

        current_time = time.time()
        
        # Check if this specific face ID exists in the dictionary and if it's on cooldown
        last_saved = self.last_saved_per_face.get(face_id, 0)
        
        if current_time - last_saved < self.interval:
            return

        point = (
            Point("FER Prediction")
            .tag("device", "pi2")
            .tag("face_id", str(face_id)) # Treat each face as a separate time-series
            .field("emotion", emotion)
            .field("confidence", float(confidence))
            .field("posture_score", float(posture_score))
            .field("posture", posture)
            .field("gesture", gesture)
        )

        try:
            self.write_api.write(
                bucket = self.bucket,
                org    = self.org,
                record = point
            )
            # Update the cooldown timer for this specific face ID
            self.last_saved_per_face[face_id] = current_time
            print(f"InfluxDB saved Face #{face_id}: {emotion} ({confidence:.2f}) | Posture: {posture} ({posture_score:.2f}) | Gesture: {gesture}")
        except Exception as e:
            print(f"InfluxDB save failed for Face #{face_id}: {e}")

    def close(self):
        self.write_api.close()
        self.client.close()