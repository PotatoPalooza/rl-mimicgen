import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("rl_mimicgen/assets/panda/panda.xml")
data = mujoco.MjData(model)

print("Opening MuJoCo viewer... Close the window to exit.")
mujoco.viewer.launch(model, data)
