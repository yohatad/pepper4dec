from setuptools import setup
from glob import glob

pkg = "behavior_controller"

setup(
    name=pkg,
    version="0.1.0",
    packages=["behavior_controller"],
    install_requires=[
        "setuptools",
        "opencv-python",
        "numpy",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Overt Attention node",
    license="",
    entry_points={
        "console_scripts": [
            "behavior_controller = behavior_controller.behavior_controller_application:main",
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
