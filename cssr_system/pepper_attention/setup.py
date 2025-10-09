from setuptools import setup
import os
from glob import glob

package_name = 'pepper_attention'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'srv'),
            glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Unified visual attention system for Pepper robot',
    license='MIT',
    tests_require=['pytest'],
   entry_points={
        'console_scripts': [
            'depth_query_server = pepper_attention.depth_query_server:main',
            'saliency_node = pepper_attention.saliency_node:main',
            'unified_attention_node = pepper_attention.unified_attention_node:main',
            'visualization_node = pepper_attention.visualization_node:main',  # NEW
        ],
    },
)