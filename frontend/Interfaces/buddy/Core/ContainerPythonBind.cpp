#include "Container.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <dlfcn.h>
namespace py = pybind11;

typedef void (*arithAddSoFuncType) (MemRef<float, 1> *, 
                                    MemRef<float, 1> *,
                                    MemRef<float, 1>*);
using OnedFloatMemRef = MemRef<float, 1>;

arithAddSoFuncType loadAddFunc() {
  void *dlHandle = dlopen("/root/xtayex-buddy-mlir/frontend/Interfaces/buddy/Core/arith_add.so", RTLD_LAZY);
  if (dlHandle) {
    arithAddSoFuncType arithAdd = reinterpret_cast<arithAddSoFuncType>(dlsym(dlHandle, "_mlir_ciface_generated_func"));
    return arithAdd;
  } else {
    throw std::runtime_error("[ERROR] Can not load shared library");
  }
}

py::list memRefGetData(OnedFloatMemRef &memref) {
  py::list ret;
  size_t size = memref.getSize();
  float *dataptr = memref.getData();
  for (size_t i = 0; i < size; i++) {
    ret.append(dataptr + i);
  }
  return ret;
}

OnedFloatMemRef createOnedFloatMemRef(py::list data, py::list sizes, intptr_t offset = 0) {
  size_t data_length = data.size();
  py::array_t<float> data_arr = py::array_t<float>(data_length);
  auto data_buffer = data_arr.request();
  float *data_ptr = static_cast<float *>(data_buffer.ptr);
  for (size_t i = 0; i < data_length; i++) {
    data_ptr[i] = py::cast<float>(data[i]);
  }

  size_t sizes_length = sizes.size();
  py::array_t<float> sizes_arr = py::array_t<float>(sizes.size());
  auto sizes_buffer = sizes_arr.request();
  intptr_t *sizes_ptr = static_cast<intptr_t *>(sizes_buffer.ptr);
  for (size_t i = 0; i < sizes_length; i++) {
    sizes_ptr[i] = py::cast<intptr_t>(sizes[i]);
  }
  return OnedFloatMemRef(data_ptr, sizes_ptr, offset);
}

PYBIND11_MODULE(PyBuddy, m) {
  py::class_<OnedFloatMemRef>(m, "MemRef1d")
    .def(py::init([](py::list data, py::list sizes, intptr_t offset = 0) {
      return createOnedFloatMemRef(data, sizes, offset);
    }))
    .def("getData", &memRefGetData);

  arithAddSoFuncType arithAddFunc = loadAddFunc();
  m.def("arith_add", arithAddFunc, "arith add operator");
}