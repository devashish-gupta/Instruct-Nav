import os
import omni.kit.test
import omni.graph.core as og
import omni.graph.core.tests as ogts
from omni.graph.core.tests.omnigraph_test_utils import _TestGraphAndNode
from omni.graph.core.tests.omnigraph_test_utils import _test_clear_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_setup_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_verify_scene


class TestOgn(ogts.OmniGraphTestCase):

    async def test_data_access(self):
        from omni.new.extension.camera_capture.ogn.camera_captureDatabase import camera_captureDatabase
        test_file_name = "camera_captureTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_OmniNewExtensionCamera_captureExtension_CameraCapture")
        database = camera_captureDatabase(test_node)
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)

        def _attr_error(attribute: og.Attribute, usd_test: bool) -> str:
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"


        self.assertTrue(test_node.get_attribute_exists("inputs:camera_name"))
        attribute = test_node.get_attribute("inputs:camera_name")
        db_value = database.inputs.camera_name
        expected_value = ""
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))
        ogts.verify_values(expected_value, db_value, _attr_error(attribute, False))

        self.assertTrue(test_node.get_attribute_exists("inputs:exec_in"))
        attribute = test_node.get_attribute("inputs:exec_in")
        db_value = database.inputs.exec_in
        expected_value = 0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))
        ogts.verify_values(expected_value, db_value, _attr_error(attribute, False))

        self.assertTrue(test_node.get_attribute_exists("outputs:exec_out"))
        attribute = test_node.get_attribute("outputs:exec_out")
        db_value = database.outputs.exec_out

        self.assertTrue(test_node.get_attribute_exists("outputs:height"))
        attribute = test_node.get_attribute("outputs:height")
        db_value = database.outputs.height

        self.assertTrue(test_node.get_attribute_exists("outputs:image_buffer"))
        attribute = test_node.get_attribute("outputs:image_buffer")
        db_value = database.outputs.image_buffer

        self.assertTrue(test_node.get_attribute_exists("outputs:width"))
        attribute = test_node.get_attribute("outputs:width")
        db_value = database.outputs.width
