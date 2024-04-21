import omni.kit.commands
from pxr import Gf
import omni.replicator.core as rep
lidar_config = "Velodyne_VLS128"

# 1. Create The Camera
_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/sensor",
    parent=None,
    config=lidar_config,
    translation=(0, 0, 1.0),
    orientation=Gf.Quatd(1,0,0,0),
)
# 2. Create and Attach a render product to the camera
render_product = rep.create.render_product(sensor.GetPath(), [1, 1])

# 3. Create a Replicator Writer that "writes" points into the scene for debug viewing
writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
writer.attach(render_product)

# 4. Create Annotator to read the data from with annotator.get_data()
annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
#annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacComputeRTXLidarPointCloud")
annotator.attach(render_product)

