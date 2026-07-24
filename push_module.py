import requests

PI5_URL = "http://10.0.30.7:3000/api/emotion/ingest"
TOKEN   = "e591962c78716e9fbd2677d2125b2375"

def push_result (emotion, confidence, frame_jpeg_bytes):
    files = {'image': ('frame.jpg', frame_jpeg_bytes, 'image/jpeg')}
    data  = {
        'emotion':emotion,
        'confidence':confidence,
        'deviceId':'pi4_emotion_cam',
        'modelVersion':'v1'
    }

    headers = {'x-emotion-token': TOKEN}

    try:
        # Added a short timeout so the video doesn't freeze if the network is slow
        r = requests.post(PI5_URL, files=files, data=data, headers=headers, timeout=2)
        print(f"Pushed to Dashboard: {r.status_code} {r.text}")
    except requests.RequestException as e:
        print(f"Dashboard push failed: {e}")
