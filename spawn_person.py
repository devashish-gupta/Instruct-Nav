import omni
from pxr import Usd, UsdGeom, Gf

# Initialize the animation people module (assumed from your description, adjust if the actual module differs)
import omni.anim.people as anim_people

def spawn_person(stage, path, position, asset_path):
    """
    Spawn a person at the given position with an animation asset.
    
    Args:
    stage (Usd.Stage): The USD stage to operate on.
    path (str): The USD path where the person will be added.
    position (tuple): A tuple (x, y, z) specifying the position in the world.
    asset_path (str): Path to the animated asset to use for the person.
    """
    # Create a transform for the person
    person_xform = UsdGeom.Xform.Define(stage, path)
    person_xform.AddTranslateOp().Set(value=Gf.Vec3f(position))

    # Use the omni.anim.people module to load and configure the person
    person = anim_people.create_person(stage, path + '/model')
    person.load_asset(asset_path)
    person.play_animation(loop=True)

    return person_xform

def main():
    # Initialize Omniverse and get the stage
    omni.usd.get_context().initialize()
    stage = omni.usd.get_context().get_stage()

    # Example: Spawning a person at the origin using a predefined animation asset
    spawn_person(stage, "/World/People/Person1", (0, 0, 0), "path/to/animation_asset.usda")

if __name__ == "__main__":
    main()
