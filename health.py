import numpy as np

class Health:
    def __init__(self, heartrate):
        self.heartrate = heartrate
    
    def setNote(self, health_note:str):
        '''
        Permits (optional) health_note from user
        ---
        self 
        health_note - Any note about the activity to be saved
        '''
        self.note = health_note 


class BikeHealth(Health):
    def __init__(self, heartrate):
        super().init(self, heartrate)

class SoccerHealth(Health):
    def __init__(self, heartrate):
        super().init(self, heartrate)
    
    def setIntervals(self, n):
        self.intervals_n = n
    