cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TXL_SOCKET_ONLY "Build without ONNX Runtime for TXL socket deployment" OFF)
set(UNITREE_SDK_ROOT "" CACHE PATH "Optional Unitree SDK install prefix")

if(NOT UNITREE_SDK_ROOT)
  set(LOCAL_UNITREE_SDK_ROOT "${PROJECT_SOURCE_DIR}/../../../../.local/unitree_robotics")
  if(EXISTS "${LOCAL_UNITREE_SDK_ROOT}/include")
    set(UNITREE_SDK_ROOT "${LOCAL_UNITREE_SDK_ROOT}")
  endif()
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

if(UNITREE_SDK_ROOT)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UNITREE_SDK_ARCH "aarch64")
  else()
    set(UNITREE_SDK_ARCH "x86_64")
  endif()

  include_directories(
    ${UNITREE_SDK_ROOT}/include
    ${UNITREE_SDK_ROOT}/thirdparty/include
    ${UNITREE_SDK_ROOT}/include/ddscxx
    ${UNITREE_SDK_ROOT}/thirdparty/include/ddscxx
  )
  link_directories(${UNITREE_SDK_ROOT}/lib)
  link_directories(${UNITREE_SDK_ROOT}/lib/${UNITREE_SDK_ARCH})
  link_directories(${UNITREE_SDK_ROOT}/thirdparty/lib/${UNITREE_SDK_ARCH})
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../include/
)

if(NOT TXL_SOCKET_ONLY)
  include_directories(
    ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  )
else()
  add_compile_definitions(TXL_SOCKET_ONLY)
endif()

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
)

if(NOT TXL_SOCKET_ONLY)
  link_libraries(
    ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
  )
endif()

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
