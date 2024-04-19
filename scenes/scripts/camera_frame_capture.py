# This script is executed the first time the script node computes, or the next time
# it computes after this script is modified or the 'Reset' button is pressed.
#
# The following callback functions may be defined in this script:
#     setup(db): Called immediately after this script is executed
#     compute(db): Called every time the node computes (should always be defined)
#     cleanup(db): Called when the node is deleted or the reset button is pressed
# Available variables:
#    db: og.Database The node interface, attributes are exposed like db.inputs.foo
#                    Use db.log_error, db.log_warning to report problems.
#    og: The omni.graph.core module

from omni.isaac.sensor.scripts.camera import get_all_camera_objects
import matplotlib.pyplot as plt


def setup(db: og.Database):
	# finding the jetbot camera object
	camera = None
	for cam in get_all_camera_objects():
		if cam.name == 'jetbot_camera':
			camera = cam
			
	db.internal_state.__dict__['camera'] = camera
	


def cleanup(db: og.Database):
	print('cleanup done')


def compute(db: og.Database):
	camera = db.internal_state.camera
	print(camera.get_rgb())
	# print(camera.get_rgba()[:, :, :3])
	