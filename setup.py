from setuptools import find_packages, setup

package_name = 'object_detector_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leev',
    maintainer_email='leev@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'aruco_prec_land = {package_name}.node:main',
            f'aruco_coordinates_pub = {package_name}.coordinates_publisher:main',
            f'aruco_infred_landing_node = {package_name}.aruco_infred_landing_node:main',
        ],
    },
)
