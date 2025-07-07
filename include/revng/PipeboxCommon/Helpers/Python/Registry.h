#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "nanobind/nanobind.h"

namespace revng::pypeline::helpers::python {

struct BaseClasses {
  nanobind::object BaseContainer;
  nanobind::object BaseAnalysis;
  nanobind::object BasePipe;
};

/// Registry class that allows registering functions that will modify the
/// provided Python module by adding Container, Analysis and Pipes subclasses.
class RegistryImpl {
private:
  using PythonModuleInitializer = void (*)(nanobind::module_ &, BaseClasses &);
  std::vector<PythonModuleInitializer> ModuleInitializers;

public:
  RegistryImpl() = default;
  ~RegistryImpl() = default;
  RegistryImpl(const RegistryImpl &) = delete;
  RegistryImpl &operator=(const RegistryImpl &) = delete;
  RegistryImpl(RegistryImpl &&) = delete;
  RegistryImpl &operator=(RegistryImpl &&) = delete;

public:
  void registerModuleInitializer(PythonModuleInitializer PMI) {
    ModuleInitializers.push_back(PMI);
  }

  void callAll(nanobind::module_ &M, BaseClasses &BC) {
    for (auto &Element : ModuleInitializers)
      Element(M, BC);
  }
};

inline RegistryImpl Registry;

} // namespace revng::pypeline::helpers::python
