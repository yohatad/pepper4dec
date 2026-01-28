from setuptools import setup
from glob import glob

pkg = "animate_behavior"

setup(
    name=pkg,
    version="0.1.0",
    packages=["animate_behavior"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohatad000@gmail.com",
    description="Animate behavior node",
    license="",
    entry_points={
        "console_scripts": [
            "animate_behavior = animate_behavior.animate_behavior_application:main",
            "animate_diagnostic = animate_behavior.animate_behavior_diagnosis:main",
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
