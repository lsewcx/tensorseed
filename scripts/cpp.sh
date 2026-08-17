conan profile detect --force
cd cpp
conan install . --output-folder=build --build=missing
cmake --preset conan-default
cmake --build --preset conan-release