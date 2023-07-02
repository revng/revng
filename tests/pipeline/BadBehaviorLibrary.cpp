/// \file BadBehaviorLibrary.cpp
/// This file, when compiled into a library, crashes in a manner configured in
/// the environment variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cassert>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "revng/Support/Assert.h"

// Ugly way to bypass the check-conventions check that enforces the usage of
// revng_abort(). We specifically don't want to use it in this case.
#define sneaky_abort abort

int *SomeGlobalPointerNobodyWillInitialize;

static void printAndFlush(std::string_view String) {
  std::cout << String << std::endl;
  std::cout.flush();
}

static void doCrash() {
  const char *CEnv = std::getenv("REVNG_CRASH_SIGNAL");
  if (CEnv == nullptr) {
    // But would this be a bug or a feature?
    revng_abort("$REVNG_CRASH_SIGNAL not set");
  }
  std::string Env(CEnv);
  int Signal = std::stoi(Env);

  switch (Signal) {
  case SIGILL:
    printAndFlush("SIGILL via illegal instruction");
    asm(".byte 0x0f, 0x0b");
    break;
  case SIGABRT:
    printAndFlush("SIGABRT via abort");
    sneaky_abort();
  case SIGSEGV:
    printAndFlush("SIGSEGV via write to uninitialized pointer");
    *SomeGlobalPointerNobodyWillInitialize = 69;
    break;
  default:
    std::cout << "Signal " << Signal << " via raise()" << std::endl;
    std::cout.flush();
    std::raise(Signal);
    break;
  }
  revng_abort("This shouldn't be executed");
}

class WillCrash {
public:
  WillCrash() { doCrash(); }
};

static WillCrash WillCrashInstance;
