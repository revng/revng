/// \file Debug.cpp
/// \brief Implementation of the debug framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/IR/Value.h"

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/revng.h"

namespace cl = llvm::cl;

size_t MaxLoggerNameLength = 0;
LogTerminator DoLog;
llvm::ManagedStatic<LoggersRegistry> Loggers;

std::ostream &dbg(std::cerr);

Logger<> PassesLog("passes");
Logger<> ReleaseLog("release");

template<bool X>
void Logger<X>::emit() {
  if (X && Enabled) {
    std::string Pad = std::string(MaxLoggerNameLength - Name.size(), ' ');
    dbg << "[" << Name.data() << Pad << "] ";
    for (unsigned I = 0; I < IndentLevel; I++)
      dbg << "  ";
    dbg << Buffer.str() << "\n";
    Buffer.str("");
    Buffer.clear();
  }
}

static std::unique_ptr<cl::list<PlaceholderEnum>> DebugLogging;
static std::unique_ptr<cl::alias> DebugLoggingAlias;

llvm::ManagedStatic<DebugLogOptionWrapper> DebugLogOption;

template<bool X>
unsigned Logger<X>::IndentLevel;

// Force instantiation
template class Logger<true>;
template class Logger<false>;
