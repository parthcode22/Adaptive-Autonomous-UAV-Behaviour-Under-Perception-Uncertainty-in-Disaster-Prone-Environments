class SensorManager:

    def __init__(self):
        self.current_sensor = "RGB"

    def choose_sensor(self, confidence):

        if confidence > 0.75:
            self.current_sensor = "RGB"

        elif confidence > 0.45:
            self.current_sensor = "IR"

        else:
            self.current_sensor = "THERMAL"

        return self.current_sensor