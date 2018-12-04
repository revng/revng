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
  llvm::raw_os_ostream Output(std::cerr);
  std::cerr << "\n";
  llvm::sys::PrintStackTrace(Output);
}

[[noreturn]] static void terminate(void) {
  print_stack_trace();
  abort();
}

static void
report(const char *Type, const char *File, unsigned Line, const char *What) {
  fprintf(stderr, "%s at %s:%d", Type, File, Line);
  if (What != NULL)
    fprintf(stderr, ": %s", What);
  fprintf(stderr, "\n");
}

void revng_assert_fail(const char *AssertionBody,
                       const char *Message,
                       const char *File,
                       unsigned Line) {
  report("Assertion failed", File, Line, Message);
  fprintf(stderr, "\n%s\n", AssertionBody);
  terminate();
}

void revng_check_fail(const char *CheckBody,
                      const char *Message,
                      const char *File,
                      unsigned Line) {
  report("Check failed", File, Line, Message);
  fprintf(stderr, "\n%s\n", CheckBody);
  terminate();
}

void revng_do_abort(const char *Message, const char *File, unsigned Line) {
  report("Abort", File, Line, Message);
  terminate();
}
