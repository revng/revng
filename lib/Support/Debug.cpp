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

// Local libraries includes
#include "revng/Support/CommandLine.h"
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

/// \brief Class for dynamically registering arguments options
template<class DataType>
class DynamicValuesClass {
private:
  struct Alternative {
    const char *Name;
    int Value;
    const char *Description;
  };

  std::vector<Alternative> Values;

public:
  void addOption(const char *Name, int Value, const char *Description) {
    Values.push_back({ Name, Value, Description });
  }

  template<class Opt>
  void apply(Opt &O) const {
    for (const Alternative &A : Values)
      O.getParser().addLiteralOption(A.Name, A.Value, A.Description);
  }
};

enum PlaceholderEnum {};
static std::unique_ptr<cl::list<PlaceholderEnum>> DebugLogging;
static std::unique_ptr<cl::alias> DebugLoggingAlias;

void LoggersRegistry::registerArguments() const {
  DynamicValuesClass<std::string> Values;
  unsigned I = 0;
  for (Logger<true> *L : Loggers)
    Values.addOption(L->name().data(), I++, L->description().data());
  auto *Opt = new cl::list<PlaceholderEnum>("debug-log",
                                            cl::desc("enable verbose logging"),
                                            Values,
                                            cl::cat(MainCategory));
  DebugLogging.reset(Opt);
  DebugLoggingAlias.reset(new cl::alias("d",
                                        cl::desc("Alias for -debug-log"),
                                        cl::aliasopt(*DebugLogging),
                                        cl::cat(MainCategory)));
}

void LoggersRegistry::activateArguments() {
  for (unsigned I : *DebugLogging)
    Loggers[I]->enable();
}

template<bool X>
unsigned Logger<X>::IndentLevel;

// Force instantiation
template class Logger<true>;
template class Logger<false>;
