from influxdb_client import InfluxDBClient

URL = "http://10.0.30.7:8086"
TOKEN = "v1-sLO515RunAvsvTZZ8PLPP13L05Yvp2CyT8OeMfd1rqPtywjqRHKq6EjQQa6N4EYuSN6XfxLYwwsiCzxgKKw=="
ORG = "EmoSys"
BUCKET = "emotionDB"

print(f"Attempting to connect to {URL}...")

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)

try:
    health = client.health()
    if health.status == "pass":
        print("Success: Connection Successful! The database is online.")
    else:
        print(f"Error: Database is not healthy: {health.message}")
        
    # Test authorization by checking buckets
    buckets_api = client.buckets_api()
    bucket = buckets_api.find_bucket_by_name(BUCKET)
    
    if bucket:
        print(f"Success: Authorization Successful! Found bucket '{BUCKET}'.")
    else:
        print(f"Error: Authorization Failed: Bucket '{BUCKET}' not found or token cannot read it.")

except Exception as e:
    print(f"\nError: Connection or Authorization Error:\n{e}")

finally:
    client.close()
