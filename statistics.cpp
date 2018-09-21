/// \file statistics.cpp
/// \brief Implementation of the statistics collection framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local includes
#include "statistics.h"

struct Handler {
  int Signal;
  bool Restore;
  struct sigaction OldHandler;
  struct sigaction NewHandler;
};

// Print statistics on SIGINT (Ctrl + C), SIGABRT (assertions) and SIGUSR1.
// For SIGUSR1, don't terminate program execution.
static std::array<Handler, 3> Handlers = { { { SIGINT, true, {}, {} },
                                             { SIGABRT, true, {}, {} },
                                             { SIGUSR1, false, {}, {} } } };

llvm::ManagedStatic<OnQuitRegistry> OnQuitStatistics;

static void onQuit() {
  dbg << "\n";
  OnQuitStatistics->dump();
}

static void onQuitSignalHandler(int Signal) {
  Handler *SignalHandler = nullptr;
  for (Handler &H : Handlers)
    if (H.Signal == Signal)
      SignalHandler = &H;

  // Assert we were notified of the signal we expected
  assert(SignalHandler != nullptr);

  onQuit();

  if (not SignalHandler->Restore)
    return;

  int Result = sigaction(Signal, &SignalHandler->OldHandler, nullptr);
  assert(Result == 0);
  raise(Signal);
}

void OnQuitRegistry::install() {
  // Dump on normal exit
  std::atexit(onQuit);

  // Register signal handlers
  for (Handler &H : Handlers) {
    H.NewHandler.sa_handler = &onQuitSignalHandler;

    int Result = sigaction(H.Signal, &H.NewHandler, &H.OldHandler);
    assert(Result == 0);
    assert(H.OldHandler.sa_handler == nullptr);
  }
}

void RunningStatistics::onQuit() {
  dump();
  dbg << "\n";
}

OnQuitInteraface::~OnQuitInteraface() {
}
