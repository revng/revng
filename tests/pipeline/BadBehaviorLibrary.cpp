/// \file BadBehaviorLibrary.cpp
/// \brief This file, when compiled into a library, crashes in a manner
/// configured in the environment variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cassert>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

int *SomeGlobalPointerNobodyWillInitialize;

static void printAndFlush(std::string_view String) {
  std::cout << String << std::endl;
  std::cout.flush();
}

static void doCrash() {
  const char *CEnv = std::getenv("REVNG_CRASH_SIGNAL");
  if (CEnv == nullptr)
    abort(); // But would this be a bug or a feature?
  std::string Env(CEnv);
  int Signal = std::stoi(Env);

  switch (Signal) {
  case SIGILL:
    printAndFlush("SIGILL via illegal instruction");
    asm(".byte 0x0f, 0x0b");
    break;
  case SIGABRT:
    printAndFlush("SIGABRT via abort()");
    abort();
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
  std::cerr << "This shouldn't be executed" << std::endl;
  abort();
}

class WillCrash {
public:
  WillCrash() { doCrash(); }
};

static WillCrash WillCrashInstance;
