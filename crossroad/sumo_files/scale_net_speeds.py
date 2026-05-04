#!/usr/bin/env python3
"""
Scale all speed attributes in a SUMO .net.xml file by a constant factor.

Targets:
  - speed="X"        on every <type> and <lane> element
  - limitTurnSpeed   attribute on the root <net> element

Usage:
  python3 scale_net_speeds.py [net_file] [factor]

Defaults:
  net_file = crossroad.net.xml  (same directory as this script)
  factor   = 0.45
"""

import xml.etree.ElementTree as ET
import shutil
import sys
import os

DEFAULT_FACTOR   = 0.45
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NET_FILE = os.path.join(SCRIPT_DIR, "crossroad.net.xml")


def scale_speeds(net_file: str, factor: float) -> None:
    if not os.path.isfile(net_file):
        print(f"[-] File not found: {net_file}")
        sys.exit(1)

    backup = net_file + ".bak"
    shutil.copy2(net_file, backup)
    print(f"[+] Backup: {backup}")

    # Preserve XML comments (requires Python 3.8+)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree   = ET.parse(net_file, parser=parser)
    root   = tree.getroot()

    changed = 0

    # --- Root <net> attribute: limitTurnSpeed ---
    if "limitTurnSpeed" in root.attrib:
        old = float(root.attrib["limitTurnSpeed"])
        new = round(old * factor, 2)
        root.attrib["limitTurnSpeed"] = f"{new:.2f}"
        print(f"  <net> limitTurnSpeed: {old:.4f} → {new:.2f}")
        changed += 1

    # --- <type> and <lane> elements: speed attribute ---
    for tag in ("type", "lane"):
        for elem in root.iter(tag):
            if "speed" in elem.attrib:
                old = float(elem.attrib["speed"])
                new = round(old * factor, 2)
                elem.attrib["speed"] = f"{new:.2f}"
                changed += 1

    print(f"[+] {changed} speed values updated (factor={factor})")

    # Re-indent to produce readable output (Python 3.9+)
    ET.indent(tree, space="    ")

    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    print(f"[+] Saved: {net_file}")


if __name__ == "__main__":
    net_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NET_FILE
    factor   = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FACTOR
    scale_speeds(net_file, factor)
