#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

#include "revng/Support/OnQuit.h"

namespace nanobind {
class module_;
}

namespace pypeline {

namespace tracerunner {
class Registry;
}

class Registry {
private:
  using PythonCallback = std::function<void(nanobind::module_ &)>;
  std::vector<PythonCallback> PythonCallbacks;

  using TRRCallback = std::function<void(pypeline::tracerunner::Registry &)>;
  std::vector<TRRCallback> TRRCallbacks;

public:
  Registry() : PythonCallbacks(), TRRCallbacks() {
    // The registry will be populated by multiple dynamic libraries, clear it
    // before exit to guarantee that there are no reverse-initialization
    // fiascos.
    OnQuit->add([this]() { this->clear(); });
  }
  ~Registry() = default;
  Registry(const Registry &) = delete;
  Registry &operator=(const Registry &) = delete;
  Registry(Registry &&) = delete;
  Registry &operator=(Registry &&) = delete;

public:
  void registerPythonCallback(PythonCallback &&Callback) {
    PythonCallbacks.push_back(std::move(Callback));
  }

  void registerTraceRunnerCallback(TRRCallback &&Callback) {
    TRRCallbacks.push_back(std::move(Callback));
  }

  void callAll(nanobind::module_ &M) {
    for (auto &Element : PythonCallbacks)
      Element(M);
  }

  void callAll(pypeline::tracerunner::Registry &M) {
    for (auto &Element : TRRCallbacks)
      Element(M);
  }

public:
  void clear() {
    PythonCallbacks.clear();
    TRRCallbacks.clear();
  }
};

extern Registry TheRegistry;

} // namespace pypeline
