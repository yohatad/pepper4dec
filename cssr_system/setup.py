from setuptools import setup, find_packages
import os
from glob import glob

package_name = "cssr_system"

setup(
    name=package_name,
    version="0.1.0",
    # expose the two packages that live in your existing dirs
    packages=["face_detection", "overt_attention"],
    package_dir={
        "face_detection": "face_detection/src",
        "overt_attention": "overt_attention/src",
    },
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        # Install Python modules alongside executables
        ("lib/" + package_name, glob("face_detection/src/*.py")),
        ("lib/" + package_name, glob("overt_attention/src/*.py")),
        # (launch/config/models are still installed by CMake; leave them there)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="CSSR system: face detection and overt attention",
    license="",
    entry_points={
        "console_scripts": [
            # What you'll run with: ros2 run cssr_system face_detection
            "face_detection = face_detection.face_detection_application:main",
            "overt_attention = overt_attention.overt_attention_application:main",
        ],
    },
)
