cmake_minimum_required(VERSION 3.8)
project(digit_sdk)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables (camera_shm, pipeline_node, publishers)
# camera_shm: plain Python, writes to SharedMemory (no rclpy)
# pipeline_node: rclpy, ProcessingEngine in-process, writes SHM output
# raw_publisher: rclpy, publishes raw frames from SHM
# surface_publisher: rclpy, publishes depth + pointcloud from SHM
# force_publisher: rclpy, publishes force_field + force_vector from SHM
install(PROGRAMS
  ros2/camera_shm.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_shm
)

install(PROGRAMS
  ros2/pipeline_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pipeline_node
)

install(PROGRAMS
  ros2/raw_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME raw_publisher
)

install(PROGRAMS
  ros2/surface_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME surface_publisher
)

install(PROGRAMS
  ros2/force_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME force_publisher
)

# Install launch files
install(DIRECTORY
  ros2/launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install URDF and meshes (digit.xacro + digit.stl)
install(DIRECTORY
  ros2/urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# Install other directories
install(DIRECTORY
  sensors/
  DESTINATION share/${PROJECT_NAME}/sensors
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_dct_equivalence test/test_dct_equivalence.py
    TIMEOUT 30)
endif()

ament_package()
