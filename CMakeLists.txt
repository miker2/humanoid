cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

project (HUMANOID)

include(${CMAKE_SOURCE_DIR}/cmake/colours.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/add_clang_format.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

include(FetchContent)
include(FindThreads)
find_library(MUJOCO_LIBRARY mujoco)
if (NOT MUJOCO_LIBRARY)
  message(FATAL_ERROR "${Red}Unable to find MuJoCo framework.${ColourReset}")
else()
  message(STATUS "${Yellow}Found mujoco framework at ${MUJOCO_LIBRARY}${ColourReset}")
endif()

FetchContent_Declare(
  flexi_cfg
  GIT_REPOSITORY  https://github.com/miker2/flexi_config_reader.git
  GIT_TAG         fix-external
  CMAKE_ARGS -DCFG_ENABLE_TEST=OFF
  )
FetchContent_MakeAvailable(flexi_cfg)


find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)

option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
if(ENABLE_CLANG_TIDY)
  if(NOT CMAKE_CROSSCOMPILING) # adds invalid paths
    find_package(ClangTools QUIET)
    if(CLANG_TIDY_FOUND)
      set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_BIN}")
    else(CLANG_TIDY_FOUND)
      message(STATUS "${Yellow}clang-tidy static analysis won't be done.${ColourReset}")
    endif(CLANG_TIDY_FOUND)
    if(NOT CLANG_FORMAT_FOUND)
      message(STATUS "${Yellow}clang-format not found. Style checks won't be done.${ColourReset}")
    endif(NOT CLANG_FORMAT_FOUND)
  endif(NOT CMAKE_CROSSCOMPILING)
else(ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "")
endif(ENABLE_CLANG_TIDY)

if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

# Build simulator
add_executable(basic basic.cc)
target_link_libraries(basic
  mujoco::mujoco
  glfw
  Threads::Threads
  flexi_cfg::flexi_cfg)
