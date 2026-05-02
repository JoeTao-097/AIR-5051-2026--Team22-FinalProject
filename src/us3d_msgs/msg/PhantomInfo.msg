# Phantom recognition result produced by a Vision-Language Model.
#
# A VLM looks at the latest color frame (optionally annotated with
# ArUco markers and / or projected surface cloud) and returns a
# structured description of what it sees on the table.

Header header

# High-level type label, free-form but should fall into a known set,
# e.g. "knee", "abdomen", "neck", "thyroid", "carotid", "cardiac",
# "wrist", "phantom_box", "unknown". Used by instruction planner as
# a hint for the scan strategy.
string phantom_type

# Confidence in [0,1] returned by the VLM (or 0 if not provided).
float32 confidence

# Free-form natural language description of the phantom and any
# anatomical landmarks the VLM identifies. Logged for debugging
# and shown to the operator as feedback.
string description

# Suggested primary scan axis, free-form: "long", "short",
# "transverse", "sagittal", "coronal", or "" if not applicable.
string scan_axis_hint

# Suggested scan length (mm) along the chosen axis. 0 = no hint.
float32 scan_length_hint_mm

# Optional 2D bounding box of the phantom in the color image, in
# pixel coordinates [x_min, y_min, x_max, y_max]. Empty list if
# the VLM did not return one. Useful for debugging and for
# downstream fusion with the depth cloud.
float32[] bbox_image_xyxy

# Two scan endpoints in NORMALISED color-image coordinates [0, 1]:
#   [u0, v0, u1, v1]
# Filled by the instruction-aware VLM call in instruction_planner —
# the model picks the two points so the line connecting them aligns
# with the user's scan instruction (long axis / short axis / a free
# direction like "from the upper-left corner to the lower-right
# corner of the phantom"). Consumed by vlm_anchor in priority over
# bbox_image_xyxy. Empty if not provided.
float32[] endpoints_xyxy_norm

# Raw JSON payload returned by the VLM (string), for debugging and
# for consumers that want to read fields the message does not
# explicitly model. Always populated.
string raw_json
