from setuptools import setup

package_name = 'apf_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='APF planner for PX4',
    license='MIT',
    entry_points={
        'console_scripts': [
            'apf_node = apf_planner.apf_node:main',
            'apf_gui = apf_planner.apf_gui:main',
        ],
    },
)
