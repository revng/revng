/// \file OnQuit.cpp
/// \brief Implementation of the OnQuit registry that allows running operation
/// at shutdown.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <array>
#include <csignal>

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/OnQuit.h"

struct Handler {
  int Signal;
  bool Restore;
  struct sigaction OldHandler;
  struct sigaction NewHandler;
};

// Print statistics on SIGINT (Ctrl + C), SIGABRT (assertions) and SIGUSR1.
// For SIGUSR1, don't terminate program execution.
static std::array<Handler, 4> Handlers = { { { SIGINT, true, {}, {} },
                                             { SIGTERM, true, {}, {} },
                                             { SIGABRT, true, {}, {} },
                                             { SIGUSR1, false, {}, {} } } };

llvm::ManagedStatic<OnQuitRegistry> OnQuit;

static void onQuitSignalHandler(int Signal) {
  Handler *SignalHandler = nullptr;
  for (Handler &H : Handlers)
    if (H.Signal == Signal)
      SignalHandler = &H;

  // Assert we were notified of the signal we expected
  revng_assert(SignalHandler != nullptr);

  OnQuit->quit();

  if (not SignalHandler->Restore)
    return;

  int Result = sigaction(Signal, &SignalHandler->OldHandler, nullptr);
  revng_assert(Result == 0);
  raise(Signal);
}

void OnQuitRegistry::install() {
  // Register signal handlers
  for (Handler &H : Handlers) {
    H.NewHandler.sa_handler = &onQuitSignalHandler;

    int Result = sigaction(H.Signal, &H.NewHandler, &H.OldHandler);
    revng_assert(Result == 0);
  }
}

void OnQuitRegistry::quit() {
  for (std::function<void()> &Handler : Registry)
    Handler();
}
