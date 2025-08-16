from setuptools import setup
import os
from glob import glob

package_name = "cssr_system"

setup(
    name=package_name,
    version="0.1.0",
    packages=["face_detection", "overt_attention"],
    package_dir={
        "face_detection": "face_detection/src",
        "overt_attention": "overt_attention/src",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        
        # Install face_detection assets
        ("share/" + package_name + "/face_detection/launch", glob("face_detection/launch/*")),
        ("share/" + package_name + "/face_detection/config", glob("face_detection/config/*")),
        ("share/" + package_name + "/face_detection/models", glob("face_detection/models/*")),
        ("share/" + package_name + "/face_detection/data", glob("face_detection/data/*")),
        
        # Install overt_attention assets
        ("share/" + package_name + "/overt_attention/launch", glob("overt_attention/launch/*")),
        ("share/" + package_name + "/overt_attention/config", glob("overt_attention/config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yohannes Haile",
    maintainer_email="yohanneh@andrew.cmu.edu",
    description="CSSR system: face detection and overt attention",
    license="",
    entry_points={
        "console_scripts": [
            "face_detection = face_detection.face_detection_application:main",
            "overt_attention = overt_attention.overt_attention_application:main",
        ],
    },
)