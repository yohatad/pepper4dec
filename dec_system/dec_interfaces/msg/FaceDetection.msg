# FaceDetection.msg
#
# Published by: /faceDetection/data  (face_detection node)
#
# Contains per-frame detections for all faces currently visible.
# All arrays are parallel — index i describes the i-th detected face.

# Unique tracking ID for each detected face (persistent across frames)
string[] face_label_id

# Centroid position for each face
# x, y: pixel coordinates in the image frame
# z:    depth in metres (0.0 if unavailable)
geometry_msgs/Point[] centroids

# Bounding box width in pixels for each face
float32[] width

# Bounding box height in pixels for each face
float32[] height

# True if mutual gaze (eye contact) was detected for each face
bool[] mutual_gaze
