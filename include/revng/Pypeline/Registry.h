#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

namespace nanobind {
class module_;
}

class TraceRunnerRegistry;

class Registry {
private:
  using PythonCallback = std::function<void(nanobind::module_ &)>;
  std::vector<PythonCallback> PythonCallbacks;

  using TRRCallback = std::function<void(TraceRunnerRegistry &)>;
  std::vector<TRRCallback> TRRCallbacks;

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

  void callAll(TraceRunnerRegistry &M) {
    for (auto &Element : TRRCallbacks)
      Element(M);
  }
};

extern Registry TheRegistry;
