from conan import ConanFile
from conan.tools.cmake import cmake_layout


class TensorSeedRecipe(ConanFile):
    name = "tensorseed"
    settings = ("os", "compiler", "build_type", "arch")
    generators = ["CMakeDeps", "CMakeToolchain"]
    requires = ["pybind11/3.0.1"]

    def layout(self) -> None:
        cmake_layout(self)