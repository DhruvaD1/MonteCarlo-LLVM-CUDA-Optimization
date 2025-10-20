cmake_minimum_required(VERSION 3.13.4)
project(AutoTilingPass)

# Find LLVM installation
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Add the LLVM pass as a shared library
add_library(LLVMAutoTiling MODULE AutoTilingPass.cpp)

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs support core irreader analysis transformutils)
target_link_libraries(LLVMAutoTiling ${llvm_libs})

# Compiler-specific options
if(APPLE)
    set_target_properties(LLVMAutoTiling PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
elseif(UNIX)
    set_target_properties(LLVMAutoTiling PROPERTIES
        COMPILE_FLAGS "-fno-rtti"
        LINK_FLAGS "-Wl,-znodelete")
endif()

# Installation
install(TARGETS LLVMAutoTiling DESTINATION lib)

# Set output directory
set_target_properties(LLVMAutoTiling PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
