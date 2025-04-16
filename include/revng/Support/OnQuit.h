#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <functional>
#include <vector>

#include "llvm/Support/ManagedStatic.h"

enum class AdditionalSignals : uint8_t {
  None,
  USR1,
  USR2,
  USR1AndUSR2,
};

class OnQuitRegistry {
public:
private:
  struct RegistryEntry {
    std::function<void()> Handler;
    AdditionalSignals Signals = AdditionalSignals::None;
  };
  std::vector<RegistryEntry> Registry;

public:
  OnQuitRegistry() = default;
  ~OnQuitRegistry() = default;
  OnQuitRegistry(OnQuitRegistry &) = delete;
  OnQuitRegistry &operator=(const OnQuitRegistry &) = delete;
  OnQuitRegistry(OnQuitRegistry &&) = delete;
  OnQuitRegistry &operator=(OnQuitRegistry &&) = delete;

  /// Registers a cleanup Handler function which will be called upon program
  /// termination
  void add(std::function<void()> &&Handler,
           AdditionalSignals Signals = AdditionalSignals::None) {
    Registry.push_back({ std::move(Handler), Signals });
  }

  /// Register the signal handler, must be called for the handlers to be called
  void install();

  /// Called at program exit, will call all registered handlers
  void quit();

private:
  static void signalHandler(int Signal);
  void callHandlersFor(int Signal);
};

extern llvm::ManagedStatic<OnQuitRegistry> OnQuit;
