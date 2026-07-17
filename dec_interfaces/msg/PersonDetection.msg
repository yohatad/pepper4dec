# PersonDetection.msg
# Message for publishing detected and tracked persons
#
# This message contains information about detected persons including:
# - Unique tracking IDs
# - Detection confidence scores
# - Centroid positions with depth (x, y in pixels, z in meters)
# - Bounding box dimensions

# Unique tracking ID for each detected person (persistent across frames)
string[] person_label_id

# Class name for each detected person (always 'person')
string[] class_names

# COCO class ID for each detected person (always 0)
int32[] class_ids

# Detection confidence score for each person (0.0 - 1.0)
float32[] confidences

# Centroid position for each person
# x, y: pixel coordinates in image frame
# z: depth in meters (0.0 if depth unavailable)
geometry_msgs/Point[] centroids

# Bounding box width in pixels for each person
float32[] width

# Bounding box height in pixels for each person
float32[] height
