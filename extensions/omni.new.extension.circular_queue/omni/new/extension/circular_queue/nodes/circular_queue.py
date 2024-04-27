"""
This is the implementation of the OGN node defined in circular_queue.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import numpy as np


class circular_queue:
    """
         This node implements a circular queue of a particular size
    """
    class State:
        def __init__(self) -> None:
            self.queue = []

    @staticmethod
    def internal_state():
        return circular_queue.State()
        
    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""
        try:
            if db.inputs.size is None or db.inputs.size == 0:
                db.log_error('Queue size cannot be zero.')

            else:
                # filling up the queue
                if len(db.internal_state.queue) < db.inputs.size:
                    db.internal_state.queue.append(db.inputs.data)
                else:
                    db.internal_state.queue.append(db.inputs.data)
                    db.internal_state.queue.pop(0)

                    # outputs
                    db.outputs.batch_data = np.concatenate(db.internal_state.queue)
                    db.outputs.buffer_lengths = np.array([len(buffer) for buffer in db.internal_state.queue])
                
        except Exception as error:
            db.log_error(str(error))
            return False

        return True
