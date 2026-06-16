
import omni.usd
import omni.physx.scripts.utils as physx_utils
from pxr import UsdPhysics, UsdGeom, PhysxSchema

stage = omni.usd.get_context().get_stage()

prim = stage.GetPrimAtPath("/World/A6")

SDF_RESOLUTION = 128  # increase for better accuracy (512, 1024), costs more memory/time

UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.MassAPI.Apply(prim)

for desc in stage.Traverse():
    if not desc.GetPath().HasPrefix(prim.GetPath()) or not desc.IsA(UsdGeom.Mesh):
        continue
    physx_utils.setCollider(desc, approximationShape="sdf")
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(desc)
    sdf_api.GetSdfResolutionAttr().Set(SDF_RESOLUTION)