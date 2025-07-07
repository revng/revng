#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

namespace nanobind {
class module_;
}

namespace pypeline {

class Registry {
private:
  using PythonModuleInitializer = void (*)(nanobind::module_ &);
  std::vector<PythonModuleInitializer> PythonModuleInitializers;

public:
  Registry() = default;
  ~Registry() = default;
  Registry(const Registry &) = delete;
  Registry &operator=(const Registry &) = delete;
  Registry(Registry &&) = delete;
  Registry &operator=(Registry &&) = delete;

public:
  void registerPythonModuleInitializer(PythonModuleInitializer &&PMI) {
    PythonModuleInitializers.push_back(std::move(PMI));
  }

  void callAll(nanobind::module_ &M) {
    for (auto &Element : PythonModuleInitializers)
      Element(M);
  }
};

extern Registry TheRegistry;

} // namespace pypeline
