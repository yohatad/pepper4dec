from setuptools import setup
from glob import glob

pkg = "text_to_speech"

setup(
    name=pkg,
    version="0.1.0",
    packages=["text_to_speech"],  
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Text to Speech node",
    license="",
    entry_points={
        "console_scripts": [
            "text_to_speech = text_to_speech.text_to_speech_application:main",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        (f"share/{pkg}/models", glob("models/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)
