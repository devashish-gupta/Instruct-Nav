"""Support for simplified access to data on nodes of type OmniNewExtensionCamera_captureExtension.CameraCapture

Captures the frames of a camera by its name in the stage
"""

import numpy
import sys
import traceback

import omni.graph.core as og
import omni.graph.core._omni_graph_core as _og
import omni.graph.tools.ogn as ogn



class camera_captureDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type OmniNewExtensionCamera_captureExtension.CameraCapture

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.camera_name
            inputs.exec_in
        Outputs:
            outputs.exec_out
            outputs.height
            outputs.image_buffer
            outputs.width
    """

    # Imprint the generator and target ABI versions in the file for JIT generation
    GENERATOR_VERSION = (1, 41, 3)
    TARGET_VERSION = (2, 139, 12)

    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}

    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata,
    #     Is_Required, DefaultValue, Is_Deprecated, DeprecationMsg
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:camera_name', 'string', 0, 'Camera Name', 'Name of the camera for which the frame should be captured', {ogn.MetadataKeys.DEFAULT: '""'}, True, "", False, ''),
        ('inputs:exec_in', 'execution', 0, 'Exec In', 'Execution in', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('outputs:exec_out', 'execution', 0, 'Exec Out', 'Execution out', {}, True, None, False, ''),
        ('outputs:height', 'int', 0, 'Height', 'Height of the captured image', {}, True, None, False, ''),
        ('outputs:image_buffer', 'float[]', 0, 'Image Buffer', 'A flattened version of the image', {ogn.MetadataKeys.DEFAULT: '[]'}, True, [], False, ''),
        ('outputs:width', 'int', 0, 'Width', 'Width of the captured image', {}, True, None, False, ''),
    ])

    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.inputs.camera_name = og.AttributeRole.TEXT
        role_data.inputs.exec_in = og.AttributeRole.EXECUTION
        role_data.outputs.exec_out = og.AttributeRole.EXECUTION
        return role_data

    class ValuesForInputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"camera_name", "exec_in", "_setting_locked", "_batchedReadAttributes", "_batchedReadValues"}
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self._batchedReadAttributes = [self._attributes.camera_name, self._attributes.exec_in]
            self._batchedReadValues = ["", 0]

        @property
        def camera_name(self):
            return self._batchedReadValues[0]

        @camera_name.setter
        def camera_name(self, value):
            self._batchedReadValues[0] = value

        @property
        def exec_in(self):
            return self._batchedReadValues[1]

        @exec_in.setter
        def exec_in(self, value):
            self._batchedReadValues[1] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _prefetch(self):
            readAttributes = self._batchedReadAttributes
            newValues = _og._prefetch_input_attributes_data(readAttributes)
            if len(readAttributes) == len(newValues):
                self._batchedReadValues = newValues

    class ValuesForOutputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"exec_out", "height", "width", "_batchedWriteValues"}
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self.image_buffer_size = 0
            self._batchedWriteValues = { }

        @property
        def image_buffer(self):
            data_view = og.AttributeValueHelper(self._attributes.image_buffer)
            return data_view.get(reserved_element_count=self.image_buffer_size)

        @image_buffer.setter
        def image_buffer(self, value):
            data_view = og.AttributeValueHelper(self._attributes.image_buffer)
            data_view.set(value)
            self.image_buffer_size = data_view.get_array_size()

        @property
        def exec_out(self):
            value = self._batchedWriteValues.get(self._attributes.exec_out)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.exec_out)
                return data_view.get()

        @exec_out.setter
        def exec_out(self, value):
            self._batchedWriteValues[self._attributes.exec_out] = value

        @property
        def height(self):
            value = self._batchedWriteValues.get(self._attributes.height)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.height)
                return data_view.get()

        @height.setter
        def height(self, value):
            self._batchedWriteValues[self._attributes.height] = value

        @property
        def width(self):
            value = self._batchedWriteValues.get(self._attributes.width)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.width)
                return data_view.get()

        @width.setter
        def width(self, value):
            self._batchedWriteValues[self._attributes.width] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _commit(self):
            _og._commit_output_attributes_data(self._batchedWriteValues)
            self._batchedWriteValues = { }

    class ValuesForState(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to state attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)

    def __init__(self, node):
        super().__init__(node)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
        self.inputs = camera_captureDatabase.ValuesForInputs(node, self.attributes.inputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        self.outputs = camera_captureDatabase.ValuesForOutputs(node, self.attributes.outputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE)
        self.state = camera_captureDatabase.ValuesForState(node, self.attributes.state, dynamic_attributes)

    class abi:
        """Class defining the ABI interface for the node type"""

        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'OmniNewExtensionCamera_captureExtension.CameraCapture'

        @staticmethod
        def compute(context, node):
            def database_valid():
                return True
            try:
                per_node_data = camera_captureDatabase.PER_NODE_DATA[node.node_id()]
                db = per_node_data.get('_db')
                if db is None:
                    db = camera_captureDatabase(node)
                    per_node_data['_db'] = db
                if not database_valid():
                    per_node_data['_db'] = None
                    return False
            except:
                db = camera_captureDatabase(node)

            try:
                compute_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:
                    return compute_function(context, node)

                db.inputs._prefetch()
                db.inputs._setting_locked = True
                with og.in_compute():
                    return camera_captureDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                stack_trace = "".join(traceback.format_tb(sys.exc_info()[2].tb_next))
                db.log_error(f'Assertion raised in compute - {error}\n{stack_trace}', add_context=False)
            finally:
                db.inputs._setting_locked = False
                db.outputs._commit()
            return False

        @staticmethod
        def initialize(context, node):
            camera_captureDatabase._initialize_per_node_data(node)
            initialize_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context, node)

            per_node_data = camera_captureDatabase.PER_NODE_DATA[node.node_id()]
            def on_connection_or_disconnection(*args):
                per_node_data['_db'] = None

            node.register_on_connected_callback(on_connection_or_disconnection)
            node.register_on_disconnected_callback(on_connection_or_disconnection)

        @staticmethod
        def release(node):
            release_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            camera_captureDatabase._release_per_node_data(node)

        @staticmethod
        def release_instance(node, target):
            camera_captureDatabase._release_per_node_instance_data(node, target)

        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False

        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata(ogn.MetadataKeys.EXTENSION, "omni.new.extension.camera_capture")
                node_type.set_metadata(ogn.MetadataKeys.UI_NAME, "Camera Capture")
                node_type.set_metadata(ogn.MetadataKeys.DESCRIPTION, "Captures the frames of a camera by its name in the stage")
                node_type.set_metadata(ogn.MetadataKeys.LANGUAGE, "Python")
                camera_captureDatabase.INTERFACE.add_to_node_type(node_type)

        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(camera_captureDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)

    NODE_TYPE_CLASS = None

    @staticmethod
    def register(node_type_class):
        camera_captureDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(camera_captureDatabase.abi, 1)

    @staticmethod
    def deregister():
        og.deregister_node_type("OmniNewExtensionCamera_captureExtension.CameraCapture")
