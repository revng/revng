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

class Registry {
private:
  using PythonCallback = std::function<void(nanobind::module_ &)>;
  std::vector<PythonCallback> PythonCallbacks;

public:
  Registry() : PythonCallbacks() {
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

  void callAll(nanobind::module_ &M) {
    for (auto &Element : PythonCallbacks)
      Element(M);
  }

private:
  void clear() { PythonCallbacks.clear(); }
};

extern Registry TheRegistry;

} // namespace pypeline
