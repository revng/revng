#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstddef>
#include <functional>
#include <vector>

#include "llvm/Support/ManagedStatic.h"

class OnQuitRegistry {
private:
  std::vector<std::function<void()>> Registry;

public:
  void install();

  /// Registers an object for having its onQuit method called upon program
  /// termination
  void add(std::function<void()> &&Handler) {
    Registry.push_back(std::move(Handler));
  }

  void quit();
};

extern llvm::ManagedStatic<OnQuitRegistry> OnQuit;
