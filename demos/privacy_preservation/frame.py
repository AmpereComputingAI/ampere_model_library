# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.humans = None # After detection
        self.people = None # After pose TODO: Better naming
        self.blurred = None
        self.pose = None
        self.detection_idx = None
        self.latency = 0.0
