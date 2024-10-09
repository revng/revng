#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <sstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"

// TODO: use a dedicated namespace
extern std::ostream &dbg;
extern size_t MaxLoggerNameLength;

#define debug_function __attribute__((used, noinline))

/// Emits \p Indentation space pairs
template<typename Stream>
void indent(Stream &Output, size_t Indentation) {
  for (size_t I = 0; I < Indentation; I++)
    Output << "  ";
}

/// Stream an instance of this class to call Logger::emit()
struct LogTerminator {
  const char *File;
  uint64_t Line;
};
#define DoLog (LogTerminator{ __FILE__, __LINE__ })

/// Logger that self-registers itself, can be disabled, has a name and follows
/// the global indentation level
///
/// The typical usage of this class is to be a static global variable in a
/// translation unit.
template<bool StaticEnabled = true>
class Logger {
private:
  static unsigned IndentLevel;

public:
  Logger(llvm::StringRef Name) : Name(Name), Enabled(false) { init(); }

  void indent(unsigned Level = 1);
  void unindent(unsigned Level = 1);
  void setIndentation(unsigned Level);

  bool isEnabled() const { return StaticEnabled && Enabled; }
  llvm::StringRef name() const { return Name; }
  // TODO: allow optional description
  llvm::StringRef description() const { return ""; }

  void enable() {
    MaxLoggerNameLength = std::max(MaxLoggerNameLength, Name.size());
    Enabled = true;
  }

  void disable() { Enabled = false; }

  /// Write a log line
  ///
  /// To call this method using the stream syntax, see LogTerminator, or simply
  /// MyLogger << DoLog;
  void flush(const LogTerminator &LineInfo = LogTerminator{ "", 0 });

  template<typename T>
  inline Logger &operator<<(const T &Other) {
    writeToLog(*this, Other, static_cast<int>(0));
    return *this;
  }

  template<bool X>
  friend void writeToLog(Logger<X> &This, const LogTerminator &T, int Ignore);

  template<bool X, typename T, typename LowPrio>
  friend void writeToLog(Logger<X> &This, const T Other, LowPrio Ignore);

  std::unique_ptr<llvm::raw_ostream> getAsLLVMStream() {
    if (Enabled)
      return std::make_unique<llvm::raw_os_ostream>(Buffer);
    return std::make_unique<llvm::raw_null_ostream>();
  }

private:
  void init();

private:
  llvm::StringRef Name;
  std::stringstream Buffer;
  bool Enabled;
};

/// Indent all loggers within the scope of this object
template<bool StaticEnabled = true>
class LoggerIndent {
public:
  LoggerIndent(Logger<StaticEnabled> &L) : L(L) { L.indent(); }
  ~LoggerIndent() { L.unindent(); }

private:
  Logger<StaticEnabled> &L;
};

/// Emit a message for the specified logger upon return
///
/// You can create an instance of this object associated to a Logger, so that
/// when the object goes out of scope (typically, on return), the emit method
/// will be invoked.
template<bool StaticEnabled = true>
class LogOnReturn {
public:
  LogOnReturn(Logger<StaticEnabled> &L) : L(L) {}
  ~LogOnReturn() { L.flush(); }

private:
  Logger<StaticEnabled> &L;
};

/// The catch-all function for logging, it can log any type not already handled
///
/// The \p Ignore argument is not used. It's only purpose is having a template
/// argument as type, which makes it less specific that other functions that
/// have a non-template type for the same argument.
///
/// In practice, you can overload this function, specifying as type of \p Ignore
/// int, and then choose the type T you want to handle.
///
/// Note that here we introduced only two levels of priority, the fall back
/// level (this function) and the ad-hoc level. Therefore, all the ad-hoc
/// overloads should never have overlapping types for \p Other.
///
/// For an example see the next specialization.
template<bool X, typename T, typename LowPrio>
inline void writeToLog(Logger<X> &This, const T Other, LowPrio) {
  if (This.isEnabled())
    This.Buffer << Other;
}

/// Specialization of writeToLog to emit a message
template<bool X>
inline void writeToLog(Logger<X> &This, const LogTerminator &LineInfo, int) {
  This.flush(LineInfo);
}

/// Specialization for llvm::StringRef
template<bool X>
inline void writeToLog(Logger<X> &This, const llvm::StringRef &S, int Ign) {
  writeToLog(This, S.str(), Ign);
}

/// Specialization for llvm::Error
template<bool X>
inline void writeToLog(Logger<X> &This, const llvm::Error &Error, int Ign) {
  std::string Message;
  {
    llvm::raw_string_ostream Stream(Message);
    Stream << Error;
  }
  writeToLog(This, Message, Ign);
}

inline std::string consumeToString(llvm::Error &&Error) {
  revng_assert(not Error.success());

  std::string Message;
  {
    llvm::raw_string_ostream Stream(Message);
    Stream << Error;
  }
  llvm::consumeError(std::move(Error));
  return Message;
}

template<typename T>
inline std::string consumeToString(llvm::Expected<T> &Expected) {
  revng_assert(not Expected);
  return consumeToString(Expected.takeError());
}

/// Enables a debug feature and disables it when goes out of scope
class ScopedDebugFeature {
public:
  /// \param Name the name of the debugging feature
  /// \param Enable whether to actually enable it or not
  ScopedDebugFeature(std::string Name, bool Enable);

  ~ScopedDebugFeature();

private:
  std::string Name;
  bool Enabled;
};

#define revng_log(Logger, Expr)  \
  do {                           \
    if (Logger.isEnabled()) {    \
      (Logger) << Expr << DoLog; \
    }                            \
  } while (0)

extern Logger<> NRALog;
extern Logger<> PassesLog;
extern Logger<> ReleaseLog;
extern Logger<> VerifyLog;

void writeToFile(llvm::StringRef What, llvm::StringRef Path) debug_function;
