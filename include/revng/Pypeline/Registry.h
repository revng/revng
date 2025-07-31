#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

namespace nanobind {
class module_;
}

namespace pypeline {

namespace tracerunner {
class Registry;
}

class Registry {
private:
  using PythonModuleInitializer = void (*)(nanobind::module_ &);
  std::vector<PythonModuleInitializer> PythonModuleInitializers;

  using TRRCallback = void (*)(pypeline::tracerunner::Registry &);
  std::vector<TRRCallback> TRRCallbacks;

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

  void registerTraceRunnerCallback(TRRCallback &&Callback) {
    TRRCallbacks.push_back(std::move(Callback));
  }

  void callAll(nanobind::module_ &M) {
    for (auto &Element : PythonModuleInitializers)
      Element(M);
  }

  void callAll(pypeline::tracerunner::Registry &M) {
    for (auto &Element : TRRCallbacks)
      Element(M);
  }
};

extern Registry TheRegistry;

} // namespace pypeline
