#include <buddy/Core/Container.h>
#include <dlfcn.h>
#include <iostream>

using namespace std;

int main() {
  void* dlHandle = dlopen("./arith_add.so", RTLD_LAZY);
  if (dlHandle) {
    typedef void (*arithAddSoFuncType)(MemRef<float, 1> *, MemRef<float, 1> *, MemRef<float, 1>*);
    arithAddSoFuncType arithAdd = reinterpret_cast<arithAddSoFuncType>(dlsym(dlHandle, "_mlir_ciface_generated_func"));
    cout << "I can load the shared library!" << endl;
    float tensorA[3] = {1.0f, 2.0f, 3.0f};
    float tensorB[3] = {3.0f, 10.0f, 1.0f};
    float tensorOut[3] = {0.0f, 0.0f, 0.0f};
    intptr_t size[1] = {3};
    MemRef<float, 1> tensorAMemRef(tensorA, size);
    MemRef<float, 1> tensorBMemRef(tensorB, size);
    MemRef<float, 1> tensorOutMemRef(tensorOut, size);
    arithAdd(&tensorOutMemRef, &tensorAMemRef, &tensorBMemRef);
    float *outData = tensorOutMemRef.getData();
    for (int i = 0; i < 3; i++) {
      cout << outData[i] << " ";
    } 
    cout << endl;
    dlclose(dlHandle);
  } else {
    cerr << "Failed to load shared library." << endl;    
  }
}
