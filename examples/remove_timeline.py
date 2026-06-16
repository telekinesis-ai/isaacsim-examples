import omni.usd
from pxr import Sdf

stage = omni.usd.get_context().get_stage()
flat_layer = stage.Flatten()

cleared = 0

def clear_anim(path):
    global cleared
    if not path.IsPropertyPath():
        return
    samples = flat_layer.ListTimeSamplesForPath(path)
    if not samples:
        return
    attr_spec = flat_layer.GetAttributeAtPath(path)
    if attr_spec is None:
        return
    # bake the first sample as a static default, then drop the animation
    attr_spec.default = flat_layer.QueryTimeSample(path, samples[0])
    attr_spec.ClearInfo("timeSamples")
    cleared += 1

flat_layer.Traverse(Sdf.Path("/"), clear_anim)

print(f"Cleared {cleared} animated attributes")
flat_layer.Export(r"D:/Telekinesis/Code/isaacsim-examples/assets/environments/vehicle_factory_01/output_static.usd")  # set a real, writable path
print("Done!")
