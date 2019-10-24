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
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Value.h"

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/revng.h"

namespace cl = llvm::cl;
using llvm::Twine;

static cl::opt<unsigned> MaxLocationLength("debug-location-max-length",
                                           cl::desc("emit file and line number "
                                                    "for log "
                                                    "messages of at most this "
                                                    "size."),
                                           cl::cat(MainCategory),
                                           cl::init(0));

size_t MaxLoggerNameLength = 0;
llvm::ManagedStatic<LoggersRegistry> Loggers;

std::ostream &dbg(std::cerr);

Logger<> PassesLog("passes");
Logger<> ReleaseLog("release");

template<bool X>
void Logger<X>::flush(const LogTerminator &LineInfo) {
  if (X && Enabled) {
    std::string Pad;

    if (MaxLocationLength != 0) {
      std::string Suffix = (Twine(":") + Twine(LineInfo.Line)).str();
      revng_assert(Suffix.size() < MaxLocationLength);
      std::string Location(LineInfo.File);
      size_t LastSlash = Location.rfind("/");
      if (LastSlash != std::string::npos)
        Location.erase(0, LastSlash + 1);

      if (Location.size() > MaxLocationLength - Suffix.size()) {
        Location.erase(MaxLocationLength - Suffix.size(), std::string::npos);
      }

      Pad = std::string(MaxLocationLength - Location.size() - Suffix.size(),
                        ' ');
      dbg << "[" << Location << Suffix << "] " << Pad;
    }

    Pad = std::string(MaxLoggerNameLength - Name.size(), ' ');
    dbg << "[" << Name.data() << "] " << Pad;
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

template<bool X>
void Logger<X>::indent(unsigned Level) {
  if (isEnabled())
    IndentLevel += Level;
}

template<bool X>
void Logger<X>::unindent(unsigned Level) {
  if (isEnabled()) {
    revng_assert(IndentLevel - Level >= 0);
    IndentLevel -= Level;
  }
}

template<bool X>
void Logger<X>::setIndentation(unsigned Level) {
  if (isEnabled())
    IndentLevel = Level;
}

// Force instantiation
template class Logger<true>;
template class Logger<false>;
