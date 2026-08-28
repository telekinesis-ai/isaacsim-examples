"""Inspect a USD file and recursively list all other USD files it depends on
(sublayers, references, payloads). Requires the `pxr` USD Python bindings
(available inside Isaac Sim's Python, or via `pip install usd-core`).

Usage:
    python3 inspect_usd_dependencies.py [path/to/file.usd]

If no path is given, defaults to the example asset shipped in this repo.
"""

import argparse
import os
import sys

from pxr import Sdf, UsdUtils

DEFAULT_USD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "environments",
    "automotive_assembly",
    "automotive_assembly.usd",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_file", nargs="?", default=DEFAULT_USD, help="Path to the root .usd/.usda file to inspect")
    args = parser.parse_args()

    layer = Sdf.Layer.FindOrOpen(args.usd_file)
    if layer is None:
        print(f"[ERROR] Could not open layer: {args.usd_file}")
        sys.exit(1)

    print(f"Root layer: {layer.identifier}\n")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(layer.identifier)

    dependent_layers = sorted({l.identifier for l in layers if l.identifier != layer.identifier})

    print(f"Found {len(dependent_layers)} dependent USD layer(s):")
    for identifier in dependent_layers:
        print(f"  - {identifier}")

    if assets:
        print(f"\nReferenced non-layer assets ({len(assets)}):")
        for asset in sorted(assets):
            print(f"  - {asset}")

    if unresolved:
        print(f"\n[WARNING] {len(unresolved)} unresolved path(s):")
        for path in sorted(unresolved):
            print(f"  - {path}")

    print("\n" + "=" * 60)
    print("Dependent USD files:")
    print("=" * 60)
    if dependent_layers:
        for identifier in dependent_layers:
            print(identifier)
    else:
        print("(none — this file has no external USD dependencies)")


if __name__ == "__main__":
    main()
