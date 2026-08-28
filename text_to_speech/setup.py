from setuptools import setup
from glob import glob

pkg = "text_to_speech"

setup(
    name=pkg,
    version="0.1.0",
    packages=["text_to_speech"],
    install_requires=["setuptools"],
    # colcon picks its pytest runner only when setup.py declares this;
    # the <test_depend>python3-pytest</test_depend> in package.xml is not
    # read by the Python test task, and without it colcon silently falls
    # back to "python -m unittest", which collects none of these tests.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Text to Speech node",
    license="",
    scripts=["scripts/text_to_speech"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        (f"share/{pkg}/models", glob("models/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)
