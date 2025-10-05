cmake_minimum_required(VERSION 3.21)
project(wombifier C)

set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")

if(NOT DEFINED MAX_SDK_PATH)
  set(MAX_SDK_PATH "$ENV{HOME}/max-sdk")
endif()

# Evita path /externals/ alla radice
set(C74_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

include("${MAX_SDK_PATH}/source/max-sdk-base/script/max-pretarget.cmake")

set(SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/source/wombifier~.c
)

add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "wombifier~")

# Include dirs per Makefiles/Ninja
if(DEFINED C74_INCLUDES)
  target_include_directories(${PROJECT_NAME} PRIVATE ${C74_INCLUDES})
endif()

set(_c74support "")
if(EXISTS "${MAX_SDK_PATH}/source/c74support")
  set(_c74support "${MAX_SDK_PATH}/source/c74support")
elseif(EXISTS "${MAX_SDK_PATH}/source/max-sdk-base/c74support")
  set(_c74support "${MAX_SDK_PATH}/source/max-sdk-base/c74support")
endif()
if(_c74support)
  target_include_directories(${PROJECT_NAME} PRIVATE
    "${_c74support}/max-includes"
    "${_c74support}/msp-includes"
    "${_c74support}/jit-includes"
  )
endif()

include("${MAX_SDK_PATH}/source/max-sdk-base/script/max-posttarget.cmake")
