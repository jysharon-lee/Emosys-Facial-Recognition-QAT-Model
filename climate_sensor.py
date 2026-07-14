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
        
        self.running = False
        self.thread = None
        
        # Try to auto-connect to the first available COM port
        self._connect()

    def _connect(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("[ClimateSensor] No COM ports found. Running in offline/mock mode.")
            return

        # Attempt connection to the first port (user can hardcode this later if needed)
        try:
            target_port = ports[0].device
            print(f"[ClimateSensor] Attempting connection to {target_port}...")
            self.ser = serial.Serial(target_port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for Arduino to reset/initialize
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

    def _update_loop(self):
        while self.running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8').strip()
                        # Expected format from Arduino: "T:24.5 H:60.2"
                        self._parse_data(line)
                except Exception as e:
                    print(f"[ClimateSensor] Error reading serial: {e}")
            else:
                # Mock Data if no sensor is plugged in (for debugging)
                # self.current_temp = 24.0
                # self.current_humidity = 55.0
                time.sleep(1)
            time.sleep(0.1)

    def _parse_data(self, line):
        """
        Parses the incoming serial string. 
        Adjust this logic based on exactly what your Arduino `Serial.println()` outputs!
        Assuming format: "Temp: 24.5C  Hum: 60%" or "24.5,60.0"
        """
        try:
            # Simple CSV parsing (e.g. "24.5,60.2")
            if "," in line:
                parts = line.split(",")
                self.current_temp = float(parts[0].strip())
                self.current_humidity = float(parts[1].strip())
            # Or Key-Value parsing (e.g. "T:24.5 H:60.2")
            elif "T:" in line and "H:" in line:
                t_str = line.split("T:")[1].split()[0]
                h_str = line.split("H:")[1].split()[0]
                self.current_temp = float(t_str)
                self.current_humidity = float(h_str)
        except Exception as e:
            pass # Ignore malformed serial reads which happen occasionally

    def get_readings(self):
        """Returns tuple of (Temperature, Humidity). Can return (None, None) if unavailable."""
        return self.current_temp, self.current_humidity

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
