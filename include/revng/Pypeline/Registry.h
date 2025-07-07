#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

namespace nanobind {
class module_;
}

class Registry {
private:
  using PythonCallback = std::function<void(nanobind::module_ &)>;
  std::vector<PythonCallback> PythonCallbacks;

public:
  void registerPythonCallback(PythonCallback &&Callback) {
    PythonCallbacks.push_back(std::move(Callback));
  }

  void callAll(nanobind::module_ &M) {
    for (auto &Element : PythonCallbacks)
      Element(M);
  }
};

extern Registry TheRegistry;
