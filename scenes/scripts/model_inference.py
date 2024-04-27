import random
import numpy as np

def setup(db):
    # setup the pytorch model
    pass

def cleanup(db):
    pass

def compute(db):
    # ignoring inputs until observation queues are full


    # decoding image data
    image_data = db.inputs.image_data
    image_batches = np.split(image_data, db.inputs.obs_horizon)
    width, height = db.inputs.image_width, db.inputs.image_height
    images = [buffer.reshape(width, height, 3) for buffer in image_batches] # assuming 3 channels
    images = np.stack(images) # model expects image batch shape: (batch, width, height, 3)
    # print(f'image shape: {images.shape}')
    
    # decoding point cloud data
    pc_data = db.inputs.point_cloud_data
    counts = db.inputs.point_counts
    min_count = np.min(counts // 3)
    cum_counts = np.cumsum(counts)
    pc_batches = np.split(pc_data, cum_counts)[:-1]
    pcs = [buffer.reshape(3, -1) for buffer in pc_batches]
    pcs = [pc[:, np.random.choice(pc.shape[1], min_count, replace=False)] for pc in pcs] # downsampling every pc to minimum point count
    pcs = np.stack(pcs) # model expects pc batch shape: (batch, 3, num_points)
    # print(f'point cloud shape: {pcs.shape}')
    

    # model = db.internal_state.model
    # linear, angular = model(image, )

    # Generate random values for linear velocity and angular velocity
    linear_velocity = random.uniform(-1.0, 1.0)  # Example range: -1.0 to 1.0
    angular_velocity = random.uniform(-1.0, 1.0)  # Example range: -1.0 to 1.0
   
    
    # Set the output attributes
    db.outputs.linear_velocity = 0.05 #linear_velocity
    db.outputs.angular_velocity = 0.01 #angular_velocity

    return True
   