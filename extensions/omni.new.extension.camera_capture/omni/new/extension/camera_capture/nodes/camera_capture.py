"""
This is the implementation of the OGN node defined in camera_capture.ogn
Copyright: Devashish Gupta, 2024
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
from omni.isaac.sensor.scripts.camera import get_all_camera_objects, Camera
import numpy as np


class camera_capture:
    """
    Captures the frames of a camera by its name in the stage
    """
    class State:
        def __init__(self) -> None:
            self.camera = -1

    @staticmethod
    def internal_state():
        return camera_capture.State()
    
    def get_initialized_camera(name) -> Camera:
        # finding the camera
        camera = None
        for cam in get_all_camera_objects():
            if cam.name == name:
                camera = cam
            
        # initializing the camera so that the rgb annotator is populated
        if camera is not None:
            camera.initialize()
            return camera
        else:
            return None

    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""
        try:
            # finding the saving the camera for later use
            if not hasattr(db.internal_state, 'camera') or db.internal_state.camera == -1:
                db.internal_state.camera = camera_capture.get_initialized_camera(name=db.inputs.camera_name)
                if db.internal_state.camera is None:
                    db.log_error('Could not find a camera with the given name.')
            else:
                # getting the latest camera frame
                image = db.internal_state.camera.get_rgb()
                
                if len(image.shape) == 3:
                    width, height, _ = image.shape
                    image_buffer = image.flatten().astype(np.float32) / 255.0

                    # returning the flattened image
                    db.outputs.image_buffer = image_buffer
                    db.outputs.width = width
                    db.outputs.height = height

        except Exception as error:
            db.log_error(str(error))
            return False

        return True
