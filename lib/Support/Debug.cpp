/// \file Debug.cpp
/// Implementation of the debug framework.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
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

/// A global registry for all the loggers
///
/// Loggers are usually global static variables in translation units, the role
/// of this class is collecting them.
class LoggersRegistry {
public:
  LoggersRegistry() {}

  void add(Logger *L) { Loggers.push_back(L); }

  size_t size() const { return Loggers.size(); }

  void enable(llvm::StringRef Name) {
    for (Logger *L : Loggers) {
      if (L->name() == Name) {
        L->enable();
        return;
      }
    }

    revng_abort("Requested logger not available");
  }

  void disable(llvm::StringRef Name) {
    for (Logger *L : Loggers) {
      if (L->name() == Name) {
        L->disable();
        return;
      }
    }

    revng_abort("Requested logger not available");
  }

  void registerArguments() const;

private:
  std::vector<Logger *> Loggers;
};

static llvm::ManagedStatic<LoggersRegistry> Loggers;

ScopedDebugFeature::ScopedDebugFeature(std::string Name, bool Enable) :
  Name(Name), Enabled(Enable) {
  if (Enabled)
    Loggers->enable(Name);
}

ScopedDebugFeature::~ScopedDebugFeature() {
  if (Enabled)
    Loggers->disable(Name);
}

std::ostream &dbg(std::cerr);

Logger PassesLog("passes");
Logger ReleaseLog("release");
Logger VerifyLog("verify");

void Logger::flush(const LogTerminator &LineInfo) {
  if (Enabled) {
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

void Logger::init() {
  Loggers->add(this);
  if (Name.size() > 0) {
    auto &Parser = DebugLogOption->TheOption.getParser();
    Parser.addLiteralOption(Name.data(), Loggers->size(), description().data());
  }
}

unsigned Logger::IndentLevel;

void Logger::indent(unsigned Level) {
  if (isEnabled())
    IndentLevel += Level;
}

void Logger::unindent(unsigned Level) {
  if (isEnabled()) {
    revng_assert(IndentLevel - Level >= 0);
    IndentLevel -= Level;
  }
}

void Logger::setIndentation(unsigned Level) {
  if (isEnabled())
    IndentLevel = Level;
}

void writeToFile(llvm::StringRef What, llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream File(Path, EC);

  if (EC)
    revng_abort(llvm::Twine("Error opening file ", Path).str().c_str());

  File << What;
}
