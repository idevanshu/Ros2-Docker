from setuptools import setup

setup(
    name='yolo_detector',
    version='0.1.0',
    packages=['yolo_detector'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_detector.yolo_node:main',
        ],
    },
)
