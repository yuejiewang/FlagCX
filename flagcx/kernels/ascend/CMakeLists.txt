# Copyright (c) 2026 BAAI. All rights reserved.

cmake_minimum_required(VERSION 3.16)
project(flagcx_ascend_unirunner LANGUAGES C CXX)

# The host adaptor paired with this kernel requires the CANN 9.0 Runtime API
# (notably aclrtMemGetAddressRange and the mapped-host-memory interfaces).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(RUN_MODE "npu" CACHE STRING "Ascend KernelLaunch run mode")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH
    "CANN toolkit installation directory")
set(SOC_VERSION "" CACHE STRING
    "Target Ascend SoC, for example Ascend910B3")

if(NOT ASCEND_CANN_PACKAGE_PATH)
  message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH must point to a CANN toolkit")
endif()
if(NOT SOC_VERSION)
  message(FATAL_ERROR "SOC_VERSION must name the target Ascend SoC")
endif()
if(NOT RUN_MODE STREQUAL "npu")
  message(FATAL_ERROR
          "FlagCX Ascend UniRunner only supports RUN_MODE=npu")
endif()

if(EXISTS
   "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
  set(ASCENDC_CMAKE_DIR
      "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS
       "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
  set(ASCENDC_CMAKE_DIR
      "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
else()
  message(FATAL_ERROR
          "ascendc_kernel_cmake was not found under the CANN toolkit")
endif()

include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

# KernelLaunch library: keep this as a complete CMake-linked shared artifact
# so ascendc_library owns its version-specific CANN runtime link closure.
ascendc_library(flagcx_ascend_unirunner SHARED
  flagcx_ascend_unirunner_reduce.cpp
)

set_target_properties(flagcx_ascend_unirunner PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  POSITION_INDEPENDENT_CODE ON
)
