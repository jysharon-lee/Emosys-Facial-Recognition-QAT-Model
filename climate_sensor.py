import serial
import serial.tools.list_ports
import time
import threading

class ClimateReader:
    def __init__(self, baudrate=9600, timeout=1.0):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.current_temp = None
        self.current_humidity = None
        self.current_co2 = None
        self.current_voc = None
        self.current_pm = None
        
        self.running = False
        self.thread = None
        
        # Try to auto-connect to the first available COM port
        self._connect()

    def _connect(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("[ClimateSensor] No COM ports found. Running in offline/mock mode.")
            return

        # Attempt connection to the first port 
        try:
            target_port = ports[0].device
            print(f"[ClimateSensor] Attempting connection to {target_port}...")
            self.ser = serial.Serial(target_port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for serial device to reset
            print(f"[ClimateSensor] Successfully connected to {target_port}!")
        except Exception as e:
            print(f"[ClimateSensor] Connection failed: {e}")
            self.ser = None

    def start(self):
        """Starts a background thread to constantly read sensor data."""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _update_loop(self): #MOCK DATAAAAAAA
        while self.running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8').strip()
                    
                        self._parse_data(line)
                except Exception as e:
                    print(f"[ClimateSensor] Error reading serial: {e}")
            else:
                # Mock Data if no sensor is plugged in
                import random
                self.current_temp = 24.0 + random.uniform(-0.5, 0.5)
                self.current_humidity = 55.0 + random.uniform(-2.0, 2.0)
                self.current_co2 = 600.0 + random.uniform(-50, 50)
                self.current_voc = 120.0 + random.uniform(-10, 10)
                self.current_pm = 35.0 + random.uniform(-5, 5)
                time.sleep(1)
            time.sleep(0.1)

    def _parse_data(self, line):
        """
        Parses the incoming serial string. 
        Expected format: "24.5,60.2,800,150,45" 
        (Temp, Humidity, CO2, VOC, PM)
        """
        try:
            if "," in line:
                parts = line.split(",")
                if len(parts) >= 5:
                    self.current_temp = float(parts[0].strip())
                    self.current_humidity = float(parts[1].strip())
                    self.current_co2 = float(parts[2].strip())
                    self.current_voc = float(parts[3].strip())
                    self.current_pm = float(parts[4].strip())
        except Exception as e:
            pass # Ignore malformed serial reads

    def get_readings(self):
        """Returns tuple of (Temp, Hum, CO2, VOC, PM). Can return None for values if unavailable."""
        return (self.current_temp, self.current_humidity, 
                self.current_co2, self.current_voc, self.current_pm)

if __name__ == "__main__":
    # Test script
    sensor = ClimateReader()
    sensor.start()
    try:
        for _ in range(5):
            print("Current Climate:", sensor.get_readings())
            time.sleep(1)
    finally:
        sensor.stop()
