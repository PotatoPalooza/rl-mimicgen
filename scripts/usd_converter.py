import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a robosuite object (MujocoObject) into USD format.")
parser.add_argument("object_id", type=str, help="The object ID to convert, or `all` for all.")
parser.add_argument(
    "--out", type=str, default="rl_mimicgen/assets", help="The directory to store the generated USD files."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# cannot be in headless mode or it breaks
args_cli.headless = True

# suppress logs
if not hasattr(args_cli, "kit_args"):
    args_cli.kit_args = ""
args_cli.kit_args += " --/log/level=error"
args_cli.kit_args += " --/log/fileLogLevel=error"
args_cli.kit_args += " --/log/outputStreamLevel=error"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import shutil  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import omni.kit.app  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from mimicgen.models.robosuite.objects import CoffeeMachineObject, CoffeeMachinePodObject  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402, F401
from robosuite.models.arenas import Arena, EmptyArena, TableArena  # noqa: E402
from robosuite.models.grippers import PandaGripper
from robosuite.models.mounts import RethinkMount  # noqa: E402
from robosuite.models.robots import Panda
from robosuite.models.tasks import Task  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

ALL_OBJECTS: dict[str, type | tuple[type, type, type]] = {
    "coffee_machine": CoffeeMachineObject,
    "coffee_pod": CoffeeMachinePodObject,
    "table": TableArena,
    "panda": (Panda, PandaGripper, RethinkMount),
}


def localize_assets(root: ET.Element, tag_name: str, out_dir: str, asset_dir: str) -> None:
    asset_out_dir = os.path.join(out_dir, asset_dir)
    os.makedirs(asset_out_dir, exist_ok=True)
    for el in root.findall(f".//asset/{tag_name}"):
        old_path = el.get("file")
        if old_path:
            filename = os.path.basename(old_path)
            new_relative_path = os.path.join(asset_dir, filename)
            new_abs_path = os.path.join(asset_out_dir, filename)
            shutil.copy(old_path, new_abs_path)
            el.set("file", new_relative_path)


def flatten_mjcf(name: str, out_dir: str) -> tuple[str, str]:
    # create scene with object
    object_cls = ALL_OBJECTS[name]
    is_robot = False
    asset_name = name
    if isinstance(object_cls, tuple):
        asset_name = "robot0"
        # for robots, we expect tuple (robot, gripper, mount)
        robot_cls, gripper_cls, mount_cls = object_cls
        arena = EmptyArena()
        robot = robot_cls(idn=0)
        robot.add_gripper(gripper_cls())
        robot.add_mount(mount_cls())
        task = Task(mujoco_arena=arena, mujoco_robots=[robot])
    elif issubclass(object_cls, Arena):
        arena = object_cls(fname=name)
        task = Task(mujoco_arena=arena, mujoco_robots=[])
    else:
        arena = EmptyArena()
        task = Task(mujoco_arena=arena, mujoco_robots=[])
        try:
            spawned_object = object_cls(name=name)
        except TypeError:
            spawned_object = object_cls()
        task.merge_objects([spawned_object])

    # convert to flattened XML
    xml_string = task.get_xml()

    # clean up XML
    root = ET.fromstring(xml_string)
    # remove top level meshdir and texturedir
    compiler = root.find(".//compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    # remove other assets in the scene (ground, walls, etc.)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        # store object body
        obj_body = None
        for body in worldbody.findall("body"):
            print(body.get("name"))
            if (body_name := body.get("name")) and asset_name in body_name:
                obj_body = body
                break
        assert obj_body is not None
        # remove everything else
        worldbody.clear()
        if obj_body is not None:
            worldbody.append(obj_body)

    # convert absolute asset paths to local
    os.makedirs(out_dir, exist_ok=True)
    localize_assets(root, "mesh", out_dir, "meshes")
    localize_assets(root, "texture", out_dir, "textures")

    # save XML
    out_path = os.path.join(out_dir, f"{name}.xml")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Flattened {name} MCJF at {out_path}")

    # return XML path for subsequent USD conversion
    return out_path, asset_name


def convert_mjcf(mjcf_path: str, name: str, out_dir: str, asset_name: str) -> None:
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    out_dir = os.path.join(out_dir, f"{name}_usd")
    # create the converter configuration
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=out_dir,
        usd_file_name=f"{name}.usd",
        fix_base=False,
        import_sites=False,
        force_usd_conversion=True,
        make_instanceable=True,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    # print output
    print(f"Generated USD file: {mjcf_converter.usd_path}")

    # -----------------------------------------------
    # Remove worldBody and move bodies to root level
    # -----------------------------------------------
    stage = Usd.Stage.Open(mjcf_converter.usd_path)
    flat_layer = stage.Flatten()
    flat_stage = Usd.Stage.Open(flat_layer)
    base_path = Sdf.Path(f"/{name}")
    original_root_dir = base_path.AppendChild(f"{asset_name}_root")
    robot_root_dir = base_path.AppendChild(f"{asset_name}_base")
    temp_root_dir = base_path.AppendChild("temp_root")
    worldbody_path = base_path.AppendChild("worldBody")

    # Rename parent folder to avoid collisions
    if flat_stage.GetPrimAtPath(original_root_dir).IsValid():
        root_dir = f"{asset_name}_root"
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(original_root_dir, temp_root_dir)
        flat_stage.GetRootLayer().Apply(edit)
    elif flat_stage.GetPrimAtPath(robot_root_dir).IsValid():  # robot root links use _base instead of _root
        root_dir = f"{asset_name}_base"
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(robot_root_dir, temp_root_dir)
        flat_stage.GetRootLayer().Apply(edit)

    # Move all links out of temp_root
    temp_prim = flat_stage.GetPrimAtPath(temp_root_dir)
    if temp_prim.IsValid():
        edit2 = Sdf.BatchNamespaceEdit()
        for child in temp_prim.GetChildren():
            old_path = child.GetPath()
            new_path = base_path.AppendChild(child.GetName())
            edit2.Add(old_path, new_path)
        flat_stage.GetRootLayer().Apply(edit2)

    # Update joints to reference correct links
    joints_prim = flat_stage.GetPrimAtPath(base_path.AppendChild("joints"))
    if joints_prim.IsValid():
        print(f"Updating joint references for {name}...")
        for joint in joints_prim.GetChildren():
            # Get all properties (Attributes + Relationships)
            for prop in joint.GetProperties():
                # Correctly check if the property is a Relationship (pointer)
                if isinstance(prop, Usd.Relationship):
                    targets = prop.GetTargets()
                    new_targets = []
                    changed = False
                    prefix = f"/{name}/{root_dir}/"
                    for t in targets:
                        path_str = t.pathString
                        print(path_str)
                        # Looking for the old path segment
                        if path_str.startswith(prefix):
                            new_path_str = path_str.replace(prefix, f"/{name}/", 1)
                            new_targets.append(Sdf.Path(new_path_str))
                            changed = True
                        else:
                            new_targets.append(t)
                    if changed:
                        prop.SetTargets(new_targets)
                        print(f"    Fixed relationship: {joint.GetName()} -> {prop.GetName()}")

    # clean up temp_root and worldbody
    if flat_stage.GetPrimAtPath(temp_root_dir).IsValid():
        flat_stage.RemovePrim(temp_root_dir)
    if flat_stage.GetPrimAtPath(worldbody_path).IsValid():
        flat_stage.RemovePrim(worldbody_path)

    # ---------------------------------------
    # Detect and flatten single rigid bodies
    # ---------------------------------------
    base_prim = flat_stage.GetPrimAtPath(base_path)
    if base_prim.IsValid():
        all_joints = []
        for prim in Usd.PrimRange(base_prim):
            if prim.IsA(UsdPhysics.Joint):
                all_joints.append(prim)

        # Flatten if there are no joints, or a single fixed joint to world frame
        is_flat_candidate = False
        if len(all_joints) == 0:
            is_flat_candidate = True
        elif len(all_joints) == 1 and all_joints[0].IsA(UsdPhysics.FixedJoint):
            is_flat_candidate = True

        if is_flat_candidate:
            # Count the remaining body prims (ignoring standard MJCF folders)
            body_prims = []
            for child in base_prim.GetChildren():
                child_name = child.GetName()
                if child_name not in ["Looks", "joints"]:
                    body_prims.append(child)

            if len(body_prims) == 1:
                single_body = body_prims[0]
                single_body_name = single_body.GetName()

                print(
                    f"Detected single rigid body '{single_body_name}' (Joints: {len(all_joints)}). Flattening and "
                    f"renaming to '{name}'..."
                )

                # Delete unnecessary joints
                for joint in all_joints:
                    flat_stage.RemovePrim(joint.GetPath())

                # Rename original base to a temp path so we can free up the name
                temp_base_path = Sdf.Path(f"/{asset_name}_temp_delete")
                edit_rename = Sdf.BatchNamespaceEdit()
                edit_rename.Add(base_path, temp_base_path)
                flat_stage.GetRootLayer().Apply(edit_rename)

                # Move the body to the newly freed base_path
                body_path_in_temp = temp_base_path.AppendChild(single_body_name)
                edit_move_body = Sdf.BatchNamespaceEdit()
                edit_move_body.Add(body_path_in_temp, base_path)
                flat_stage.GetRootLayer().Apply(edit_move_body)

                # Move Looks into the new base_path
                looks_path_in_temp = temp_base_path.AppendChild("Looks")
                new_looks_path = base_path.AppendChild("Looks")
                if flat_stage.GetPrimAtPath(looks_path_in_temp).IsValid():
                    edit_move_looks = Sdf.BatchNamespaceEdit()
                    edit_move_looks.Add(looks_path_in_temp, new_looks_path)
                    flat_stage.GetRootLayer().Apply(edit_move_looks)

                # Clean up the temp base path (which also deletes empty 'joints')
                flat_stage.RemovePrim(temp_base_path)

                # Fix RigidBody APIs to ensure exactly one exists (on the root)
                new_root_prim = flat_stage.GetPrimAtPath(base_path)
                if new_root_prim.IsValid():
                    flat_stage.SetDefaultPrim(new_root_prim)

                    # Iterate through the root and all descendants
                    for prim in Usd.PrimRange(new_root_prim):
                        # 1. Strip ArticulationRootAPI entirely (not allowed on RigidObjects)
                        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                            print(f"    Removed ArticulationRootAPI from {prim.GetName()}")

                        # 2. Strip duplicate RigidBodyAPIs from descendants
                        if prim.GetPath() != base_path:  # Skip the root itself
                            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                                print(f"    Removed duplicate RigidBodyAPI from {prim.GetName()}")

                    # 3. Apply RigidBodyAPI to root if missing
                    if not new_root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI.Apply(new_root_prim)
                        print(f"    Applied RigidBodyAPI to root '{name}'")
            else:
                print(f"No joints found, but detected {len(body_prims)} bodies. Leaving hierarchy as-is.")
        else:
            print(f"Detected joints in {name}. Leaving as articulated hierarchy.")
    # =========================================================================

    # export to file
    flat_stage.GetRootLayer().Export(mjcf_converter.usd_path)
    print(f"Final {name} USD saved to {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)


def main() -> None:
    object_id = args_cli.object_id
    if object_id == "all":
        objects = list(ALL_OBJECTS.keys())
    else:
        if object_id not in ALL_OBJECTS:
            raise ValueError(f"'{object_id}' not in available objects. Valid keys are: {list(ALL_OBJECTS.keys())}")
        objects: list[str] = [object_id]

    out_dir = os.path.abspath(args_cli.out)
    for name in objects:
        obj_out_dir = os.path.join(out_dir, name)
        xml_path, asset_name = flatten_mjcf(name, obj_out_dir)
        convert_mjcf(xml_path, name, obj_out_dir, asset_name)


if __name__ == "__main__":
    main()
    simulation_app.close()
