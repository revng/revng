/// \file OnQuit.cpp
/// \brief Implementation of the OnQuit registry that allows running operation
/// at shutdown.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <csignal>

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/OnQuit.h"

using SignalHandlerT = decltype(signal(0, nullptr));
struct Handler {
  int Signal;
  bool Restore;
  SignalHandlerT OldHandler;
};

// Print statistics on SIGINT (Ctrl + C), SIGABRT (assertions) and SIGUSR1.
// For SIGUSR1, don't terminate program execution.
static std::array<Handler, 5> Handlers = { { { SIGINT, true, {} },
                                             { SIGTERM, true, {} },
                                             { SIGABRT, true, {} },
                                             { SIGUSR1, false, {} },
                                             { SIGUSR2, false, {} } } };

llvm::ManagedStatic<OnQuitRegistry> OnQuit;

void OnQuitRegistry::signalHandler(int Signal) {
  auto SignalHandler = llvm::find_if(Handlers, [&Signal](const Handler &H) {
    return H.Signal == Signal;
  });
  // Assert we were notified of the signal we expected
  revng_assert(SignalHandler != Handlers.end());

  OnQuit->callHandlersFor(Signal);
  if (not SignalHandler->Restore)
    return;

  // If here the signal needs to be propagated to the old handler
  SignalHandlerT Result = std::signal(Signal, SignalHandler->OldHandler);
  revng_assert(Result != SIG_ERR);
  raise(Signal);
}

void OnQuitRegistry::install() {
  // Register signal handlers
  for (Handler &H : Handlers) {
    H.OldHandler = std::signal(H.Signal, &this->signalHandler);
    revng_assert(H.OldHandler != SIG_ERR);
  }
}

void OnQuitRegistry::callHandlersFor(int Signal) {
  if (Signal == SIGUSR1 or Signal == SIGUSR2) {
    for (RegistryEntry &Entry : Registry) {
      if ((Signal == SIGUSR1 and Entry.Signals == AdditionalSignals::USR1)
          or (Signal == SIGUSR2 and Entry.Signals == AdditionalSignals::USR2)
          or (Entry.Signals == AdditionalSignals::USR1AndUSR2))
        Entry.Handler();
    }
  } else {
    for (RegistryEntry &Entry : Registry)
      Entry.Handler();
  }
}

void OnQuitRegistry::quit() {
  callHandlersFor(SIGINT);
}
