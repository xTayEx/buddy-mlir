add_subdirectory(FrontendGen)
add_subdirectory(Interfaces)

set(Python3_EXECUTABLE /root/miniconda3/bin/python3)
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
  -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE STATUS
  OUTPUT_VARIABLE PACKAGE_DIR
  ERROR_QUIET)
set(pybind11_DIR "${PACKAGE_DIR}")
find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(PyBuddy "./Interfaces/buddy/Core/ContainerPythonBind.cpp")
target_link_libraries(PyBuddy PRIVATE dl)