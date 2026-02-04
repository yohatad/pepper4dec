from setuptools import setup
import os
from glob import glob

package_name = 'tts'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the Python 2 script to lib directory
        (os.path.join('lib', package_name), ['scripts/send_and_play_audio.py']),
        # Install voice clones to share directory
        (os.path.join('share', package_name, 'voice_clones'), 
            glob('voice_clones/*.wav')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mahadanso',
    maintainer_email='mahadanso70@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tts_server = tts.tts_application:main',
        ],
    },
)
