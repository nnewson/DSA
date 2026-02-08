# BloomFilter in C++

Setup [vcpkg](https://vcpkg.io/en/) on the build machine, and ensure that VCPKG_ROOT is available in the PATH environment variable.
Details of how to do this can be found at steps 1 and 2 in this [getting started doc](https://learn.microsoft.com/en-gb/vcpkg/get_started/get-started).

Create a version of `CMakeUserPresets.json` in the same directory as this README.md.
This will need to be updated to point to the vcpkg running on the build machine.
Details of how to setup this up can been found in the link above under step 4.

Configure CMake (which will install and build dependencies via vcpkg):

```bash
cmake --preset=default
```

Build the test via CMake:

```bash
cmake --build build
```

Run the tests:

```bash
./build/test_bloom_filter
```

Build documentation into the docs folder:

```bash
doxygen Doxyfile
```
