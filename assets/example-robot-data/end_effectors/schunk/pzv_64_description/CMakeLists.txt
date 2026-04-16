cmake_minimum_required(VERSION 2.8.3)
project(pzv_64_description)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS)
catkin_package()

install(DIRECTORY config launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
