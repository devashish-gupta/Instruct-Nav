import numpy as np

def setup(db: og.Database):
    pass


def cleanup(db: og.Database):
    pass


def compute(db: og.Database):
    input_data = db.inputs.data
    x_data = np.array([p[0] for p in input_data])
    y_data = np.array([p[1] for p in input_data])
    z_data = np.array([p[2] for p in input_data])
    
    output_data = np.row_stack((x_data, y_data, z_data))
    db.outputs.float_data = output_data
    
    return True