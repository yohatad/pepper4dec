from setuptools import setup
from glob import glob

pkg = 'overt_attention'

setup(
    name=pkg,
    version='1.0.0',
    packages=[pkg],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yohannes Haile',
    maintainer_email='yohanneh@alumni.cmu.edu',
    description='Unified visual attention system for robot overt attention',
    license='MIT',
    entry_points={
        'console_scripts': [
            'saliency_node = overt_attention.saliency_node:main',
            'unified_attention_node = overt_attention.unified_attention_node:main',
            'visualization_node = overt_attention.visualization_node:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages', [f"resource/{pkg}"]),
        (f"share/{pkg}", ['package.xml']),
        (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)
