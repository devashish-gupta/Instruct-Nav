.. _OmniNewExtensionCamera_captureExtension_CameraCapture_1:

.. _OmniNewExtensionCamera_captureExtension_CameraCapture:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Camera Capture
    :keywords: lang-en omnigraph node omninewextensioncamera_captureextension camera-capture


Camera Capture
==============

.. <description>

Captures the frames of a camera by its name in the stage

.. </description>


Installation
------------

To use this node enable :ref:`omni.new.extension.camera_capture<ext_omni_new_extension_camera_capture>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Camera Name (*inputs:camera_name*)", "``string``", "Name of the camera for which the frame should be captured", ""
    "Exec In (*inputs:exec_in*)", "``execution``", "Execution in", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:exec_out*)", "``execution``", "Execution out", "None"
    "Height (*outputs:height*)", "``int``", "Height of the captured image", "None"
    "Image Buffer (*outputs:image_buffer*)", "``float[]``", "A flattened version of the image", "[]"
    "Width (*outputs:width*)", "``int``", "Width of the captured image", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "OmniNewExtensionCamera_captureExtension.CameraCapture"
    "Version", "1"
    "Extension", "omni.new.extension.camera_capture"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Camera Capture"
    "Generated Class Name", "camera_captureDatabase"
    "Python Module", "omni.new.extension.camera_capture"

