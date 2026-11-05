cmake_minimum_required(VERSION 3.18)

project(pyehm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PYEHM_BUILD_PYTHON "Build the _pyehm Python extension module" ON)
option(PYEHM_BUILD_CPP_TEST "Build the cpp_test executable" ON)
set(PYEHM_MODULE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
    "Directory where the _pyehm extension module is written")

if(MSVC)
    add_compile_options(/utf-8 /bigobj)
endif()

# ---------------------------------------------------------------------------
# Core C++ library (equivalent to the Debug/Release VS configurations)
# ---------------------------------------------------------------------------
file(GLOB EHM_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
)
file(GLOB EHM_CORE_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.h
)

add_library(ehm STATIC ${EHM_CORE_SOURCES} ${EHM_CORE_HEADERS})
target_include_directories(ehm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# C++ test executable
# ---------------------------------------------------------------------------
if(PYEHM_BUILD_CPP_TEST)
    add_executable(cpp_test cpp_test/cpp_test.cpp)
    target_link_libraries(cpp_test PRIVATE ehm)
endif()

# ---------------------------------------------------------------------------
# Python extension module (equivalent to the "Debug/Release Python" configs)
# ---------------------------------------------------------------------------
if(PYEHM_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Prefer the pybind11 installed in the selected Python environment,
    # otherwise download it.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmakedir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
    endif()
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found in ${Python_EXECUTABLE}; fetching it")
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.0.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    # Version string, as setup.py derives it from setuptools_scm
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from setuptools_scm import get_version; print(get_version())"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PYEHM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT PYEHM_VERSION)
        set(PYEHM_VERSION "0.0.0+dev")
    endif()

    pybind11_add_module(_pyehm
        src/module.cpp
        src/Docstrings.cpp
        src/Docstrings.h
    )
    target_link_libraries(_pyehm PRIVATE ehm)
    target_compile_definitions(_pyehm PRIVATE VERSION_INFO=${PYEHM_VERSION})

    # Write the module next to the pyehm package so `import pyehm` picks it up
    # (the $<1:> stops multi-config generators appending a per-config subdir).
    set_target_properties(_pyehm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${PYEHM_MODULE_OUTPUT_DIR}>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${PYEHM_MODULE_OUTPUT_DIR}>"
    )
endif()
