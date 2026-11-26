from glob import glob

from setuptools import setup

pkg = 'remote_dialogue'

setup(
    name=pkg,
    version='0.1.0',
    packages=[pkg],
    install_requires=['setuptools'],
    # colcon picks its pytest runner only when setup.py declares this (see
    # speech_event/setup.py).
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Yohannes',
    maintainer_email='yohatad123@gmail.com',
    description='Pepper client for the remote /ws/pepper ASR-LLM-TTS voice server',
    license='BSD-3-Clause',
    scripts=['scripts/remote_dialogue'],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{pkg}']),
        (f'share/{pkg}', ['package.xml']),
        (f'share/{pkg}/launch', glob('launch/*.launch.py')),
        (f'share/{pkg}/config', glob('config/*')),
    ],
)
