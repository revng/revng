/// \file Statistics.cpp
/// \brief Implementation of the statistics collection framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CommandLine.h"
#include "revng/Support/Statistics.h"

namespace cl = llvm::cl;

cl::opt<bool> Statistics("statistics",
                         cl::desc("print statistics upon exit or "
                                  "SIGINT. Use "
                                  "this argument, ignore -stats."),
                         cl::cat(MainCategory));

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

static void onQuit() {
  dbg << "\n";
  OnQuit->dump();
}

static void onQuitSignalHandler(int Signal) {
  Handler *SignalHandler = nullptr;
  for (Handler &H : Handlers)
    if (H.Signal == Signal)
      SignalHandler = &H;

  // Assert we were notified of the signal we expected
  revng_assert(SignalHandler != nullptr);

  onQuit();

  if (not SignalHandler->Restore)
    return;

  int Result = sigaction(Signal, &SignalHandler->OldHandler, nullptr);
  revng_assert(Result == 0);
  raise(Signal);
}

void OnQuitRegistry::install() {
  // TODO: can we use llvm::sys::AddSignalHandler?
  // Register signal handlers
  for (Handler &H : Handlers) {
    H.NewHandler.sa_handler = &onQuitSignalHandler;

    int Result = sigaction(H.Signal, &H.NewHandler, &H.OldHandler);
    revng_assert(Result == 0);
    // revng_assert(H.OldHandler.sa_handler == nullptr);
  }
}
