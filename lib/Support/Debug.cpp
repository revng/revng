/// \file Debug.cpp
/// \brief Implementation of the debug framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Value.h"

#include "revng/Support/Debug.h"

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
Logger<> VerifyLog("verify");

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
      dbg << "[" << Location << Suffix << Pad << "] ";
    }

    Pad = std::string(MaxLoggerNameLength - Name.size(), ' ');
    dbg << "[" << Name.data() << Pad << "] ";
    dbg << std::string(IndentLevel * 2, ' ');

    std::string Data = Buffer.str();
    if (Data.size() > 0 and Data.back() == '\n')
      Data.resize(Data.size() - 1);

    std::string Delimiter = "\n";
    size_t Start = 0;
    size_t End = Data.find(Delimiter);
    dbg << Data.substr(Start, End) << "\n";

    if (End != std::string::npos) {
      Pad = std::string(3 + MaxLoggerNameLength + IndentLevel * 2, ' ');
      do {
        Start = End + Delimiter.length();
        End = Data.find(Delimiter, Start);
        dbg << Pad << Data.substr(Start, End - Start) << "\n";
      } while (End != std::string::npos);
    }

    Buffer.str("");
    Buffer.clear();
  }
}

enum PlaceholderEnum {
};
struct DebugLogOptionList : public llvm::cl::list<PlaceholderEnum> {
  using list = llvm::cl::list<PlaceholderEnum>;
  DebugLogOptionList() :
    list("debug-log",
         llvm::cl::desc("enable verbose logging"),
         llvm::cl::cat(MainCategory)) {}

  virtual bool addOccurrence(unsigned Pos,
                             llvm::StringRef ArgName,
                             llvm::StringRef Value,
                             bool MultiArg = false) override {
    Loggers->enable(Value);
    return list::addOccurrence(Pos, ArgName, Value, MultiArg);
  }
};

struct DebugLogOptionWrapper {
  DebugLogOptionList TheOption;
};

static std::unique_ptr<cl::list<PlaceholderEnum>> DebugLogging;
static std::unique_ptr<cl::alias> DebugLoggingAlias;

llvm::ManagedStatic<DebugLogOptionWrapper> DebugLogOption;

template<>
void Logger<true>::init() {
  Loggers->add(this);
  DebugLogOption->TheOption.getParser().addLiteralOption(Name.data(),
                                                         Loggers->size(),
                                                         description().data());
}

template<>
void Logger<false>::init() {
}

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

void writeToFile(llvm::StringRef What, llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream File(Path, EC);

  if (EC)
    revng_abort(llvm::Twine("Error opening file ", Path).str().c_str());

  File << What;
}
