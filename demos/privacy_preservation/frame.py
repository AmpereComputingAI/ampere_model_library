class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.humans = None # After detection
        self.people = None # After pose TODO: Better naming
        self.blurred = None
        self.pose = None
        self.detection_idx = None
        self.init_time = None
