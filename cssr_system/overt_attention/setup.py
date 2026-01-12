from setuptools import setup
from glob import glob

pkg = "overt_attention"

setup(
    name=pkg,
    version="0.1.0",
    packages=["overt_attention"],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Yohannes",
    maintainer_email="yohanneh@alumni.cmu.edu",
    description="Overt Attention node",
    license="",
    entry_points={
        "console_scripts": [
            "overt_attention = overt_attention.overt_attention_application:main",
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
