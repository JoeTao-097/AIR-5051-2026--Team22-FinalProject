# Scan region definition from ArUco marker localization
Header header
geometry_msgs/Point origin       # scan region origin in base_link
geometry_msgs/Vector3 scan_dir   # primary scan direction (unit vector)
geometry_msgs/Vector3 lateral_dir # lateral direction (unit vector)
geometry_msgs/Vector3 normal     # surface normal (unit vector, pointing up)
float64 length                   # scan region length along scan_dir (m)
float64 width                    # scan region width along lateral_dir (m)
