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

# Install Python executables (camera_shm + process_node)
# camera_shm: plain Python, writes to SharedMemory (no rclpy)
# process_node: rclpy publisher, ProcessingEngine in-process (no rclpy dependency in processing layer)
install(PROGRAMS
  ros2/camera_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_shm
)

install(PROGRAMS
  ros2/process_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME process_node
)

install(PROGRAMS
  ros2/raw_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME raw_bridge_node
)

# Install launch files
install(DIRECTORY
  ros2/launch/
  DESTINATION share/${PROJECT_NAME}/launch
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
