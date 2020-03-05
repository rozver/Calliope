from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint


# Class for clipping Critic weights
class ClipConstraint(Constraint):

    # Initialization
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # Clip weights to range [-self.clip_value; self.clip_value]
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # Get configuration
    def get_config(self):
        return {'clip_value': self.clip_value}
