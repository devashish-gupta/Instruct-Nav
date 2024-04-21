.. _OmniNewExtensionCircular_queueExtension_CircularQueue_1:

.. _OmniNewExtensionCircular_queueExtension_CircularQueue:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Circular Queue
    :keywords: lang-en omnigraph node omninewextensioncircular_queueextension circular-queue


Circular Queue
==============

.. <description>

This node implements a circular queue of a particular size

.. </description>


Installation
------------

To use this node enable :ref:`omni.new.extension.circular_queue<ext_omni_new_extension_circular_queue>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Data (*inputs:data*)", "``float[]``", "Any data that needs to be stored within the queue", "[]"
    "Exec In (*inputs:exec_in*)", "``execution``", "Execution in", "0"
    "Size (*inputs:size*)", "``int``", "Size of the circular queue", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Batch Data (*outputs:batch_data*)", "``float[]``", "Batch of data from the past", "None"
    "Exec Out (*outputs:exec_out*)", "``execution``", "Execution out", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "OmniNewExtensionCircular_queueExtension.CircularQueue"
    "Version", "1"
    "Extension", "omni.new.extension.circular_queue"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Circular Queue"
    "Generated Class Name", "circular_queueDatabase"
    "Python Module", "omni.new.extension.circular_queue"

