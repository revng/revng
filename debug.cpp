/// \file debug.cpp
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

// Local includes
#include "debug.h"

static size_t MaxLoggerNameLength = 0;
LogTerminator DoLog;
llvm::ManagedStatic<LoggersRegistry> Loggers;

bool DebuggingEnabled = false;
std::ostream &dbg(std::cerr);
static std::vector<std::string> DebugFeatures;

bool isDebugFeatureEnabled(std::string Name) {
  return std::count(DebugFeatures.begin(), DebugFeatures.end(), Name) != 0;
}

void enableDebugFeature(std::string Name) {
  if (!isDebugFeatureEnabled(Name))
    DebugFeatures.push_back(Name);

  Loggers->enable(Name);

  if (Name.size() > MaxLoggerNameLength)
    MaxLoggerNameLength = Name.size();
}

void disableDebugFeature(std::string Name) {
  auto It = std::find(DebugFeatures.begin(),
                      DebugFeatures.end(),
                      Name);
  if (It != DebugFeatures.end())
    DebugFeatures.erase(It);

  Loggers->disable(Name);
}

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

template<bool X>
unsigned Logger<X>::IndentLevel;

// Force instantiation
template class Logger<true>;
template class Logger<false>;
