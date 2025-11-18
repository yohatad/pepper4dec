from setuptools import setup
from glob import glob

pkg = "speech_event"

setup(
    name=pkg,
    version="0.1.0",
    packages=["speech_event"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Speech recognition node",
    license="",
    entry_points={
        "console_scripts": [
            "speech_event = speech_event.speech_event_application:main",
            "speech_event_recorder  = speech_event.speech_event_recorder:main",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        # (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        (f"share/{pkg}/models", glob("models/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)