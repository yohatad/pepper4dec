# ObjectDetection.msg
# Message for publishing detected and tracked objects
#
# This message contains information about detected objects including:
# - Unique tracking IDs
# - Object class information (name and ID)
# - Detection confidence scores
# - Centroid positions with depth (x, y in pixels, z in meters)
# - Bounding box dimensions

# Unique tracking ID for each detected object (persistent across frames)
string[] object_label_id

# Class name for each detected object (e.g., 'person', 'car', 'bottle')
string[] class_names

# COCO class ID for each detected object (0-79)
int32[] class_ids

# Detection confidence score for each object (0.0 - 1.0)
float32[] confidences

# Centroid position for each object
# x, y: pixel coordinates in image frame
# z: depth in meters (0.0 if depth unavailable)
geometry_msgs/Point[] centroids

# Bounding box width in pixels for each object
float32[] width

# Bounding box height in pixels for each object
float32[] height
