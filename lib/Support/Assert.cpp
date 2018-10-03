/// \file revng-assert.cpp
/// \brief Implementation of the various functions to assert and abort.

// Standard includes
#include <cassert>
#include <iostream>

// LLVM includes
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng/Support/Assert.h"

static void print_stack_trace() {
  llvm::raw_os_ostream Output(std::cout);
  std::cout << "\n";
  llvm::sys::PrintStackTrace(Output);
}

[[noreturn]] static void terminate(void) {
  print_stack_trace();
  abort();
}

static void
report(const char *Type, const char *File, unsigned Line, const char *What) {
  fprintf(stderr, "%s at %s:%d: %s\n", Type, File, Line, What);
}

void revng_assert_fail(const char *AssertionBody,
                       const char *Message,
                       const char *File,
                       unsigned Line) {
  report("Assertion failed", File, Line, Message);
  fprintf(stderr, "%s\n", AssertionBody);
  terminate();
}

void revng_check_fail(const char *CheckBody,
                      const char *Message,
                      const char *File,
                      unsigned Line) {
  report("Check failed", File, Line, Message);
  fprintf(stderr, "%s\n", CheckBody);
  terminate();
}

void revng_do_abort(const char *Message, const char *File, unsigned Line) {
  report("Abort", File, Line, Message);
  terminate();
}
