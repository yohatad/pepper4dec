from setuptools import setup
from glob import glob

pkg = "gesture_execution"

setup(
    name=pkg,
    version="0.1.0",
    packages=["gesture_execution"],
    install_requires=[
        "setuptools",
        "opencv-python",
        "numpy",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Gesture Execution node",
    license="",
    entry_points={
        "console_scripts": [
            "gesture_execution = gesture_execution.gesture_execution_application:main",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        # (f"share/{pkg}/models", glob("models/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)
