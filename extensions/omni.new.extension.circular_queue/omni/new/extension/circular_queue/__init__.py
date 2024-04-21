
import omni.ext

# Any class derived from `omni.ext.IExt` in a top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when the extension is enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() will be called.
class OmniNewExtensionCircular_queueExtension(omni.ext.IExt):
    # ext_id is the current extension id. It can be used with the extension manager to query additional information,
    # such as where this extension is located in the filesystem.
    def on_startup(self, ext_id):
        print("[omni.new.extension.circular_queue] OmniNewExtensionCircular_queueExtension startup", flush=True)

    def on_shutdown(self):
        print("[omni.new.extension.circular_queue] OmniNewExtensionCircular_queueExtension shutdown", flush=True)
