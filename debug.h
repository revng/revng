#ifndef DEBUG_H
#define DEBUG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <functional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"

// Local includes
#include "revng-assert.h"

// TODO: use a dedicated namespace
extern bool DebuggingEnabled;
extern std::ostream &dbg;

#define debug_function __attribute__((used, noinline))

bool isDebugFeatureEnabled(std::string Name);
void enableDebugFeature(std::string Name);
void disableDebugFeature(std::string Name);

/// \brief Executes \p code only if \p feature is enabled
///
/// \note This approach is deprecated, please consider using Logger and letting
///       the DCE do its job for you.
#define DBG(feature, code)                                    \
  do {                                                        \
    if (DebuggingEnabled && isDebugFeatureEnabled(feature)) { \
      code;                                                   \
    }                                                         \
  } while (0)

/// \brief Enables a debug feature and disables it when goes out of scope
class ScopedDebugFeature {
public:
  /// \param Name the name of the debugging feature
  /// \param Enable whether to actually enable it or not
  ScopedDebugFeature(std::string Name, bool Enable) :
    Name(Name),
    Enabled(Enable) {
    if (Enabled)
      enableDebugFeature(Name);
  }

  ~ScopedDebugFeature() {
    if (Enabled)
      disableDebugFeature(Name);
  }

private:
  std::string Name;
  bool Enabled;
};

/// \brief Stream an instance of this class to call Logger::emit()
struct LogTerminator {};
extern LogTerminator DoLog;

/// \brief Logger that self-registers itself, can be disable, has a name and
///        follows the global indentation level
///
/// The typical usage of this class is to be a static global variable in a
/// translation unit.
template<bool StaticEnabled = true>
class Logger {
private:
  static unsigned IndentLevel;

public:
  Logger(llvm::StringRef Name) : Name(Name), Enabled(false) { init(); }

  void indent(unsigned Level = 1) {
    if (isEnabled())
      IndentLevel += Level;
  }

  void unindent(unsigned Level = 1) {
    if (isEnabled()) {
      revng_assert(IndentLevel - Level >= 0);
      IndentLevel -= Level;
    }
  }

  void setIndentation(unsigned Level) {
    if (isEnabled())
      IndentLevel = Level;
  }

  bool isEnabled() const { return StaticEnabled && Enabled; }
  llvm::StringRef name() const { return Name; }

  void enable() { Enabled = true; }

  void disable() { Enabled = false; }

  /// \brief Write a log line
  ///
  /// To call this method using the stream syntax, see LogTerminator, or simply
  /// MyLogger << DoLog;
  void emit();

  template<typename T>
  inline Logger &operator<<(const T Other) {
    writeToLog(*this, Other, static_cast<int>(0));
    return *this;
  }

  template<bool X>
  friend void writeToLog(Logger<X> &This, const LogTerminator T, int Ignore);

  template<bool X, typename T, typename LowPrio>
  friend void writeToLog(Logger<X> &This, const T Other, LowPrio Ignore);

private:
  void init();

private:
  llvm::StringRef Name;
  std::stringstream Buffer;
  bool Enabled;
};

/// \brief Indent all loggers within the scope of this object
template<bool StaticEnabled = true>
class LoggerIndent {
public:
  LoggerIndent(Logger<StaticEnabled> &L) : L(L) { L.indent(); }
  ~LoggerIndent() { L.unindent(); }

private:
  Logger<StaticEnabled> &L;
};

/// \brief Emit a message for the specified logger upon return
///
/// You can create an instance of this object associated to a Logger, so that
/// when the object goes out of scope (typically, on return), the emit method
/// will be invoked.
template<bool StaticEnabled = true>
class LogOnReturn {
public:
  LogOnReturn(Logger<StaticEnabled> &L) : L(L) {}
  ~LogOnReturn() { L.emit(); }

private:
  Logger<StaticEnabled> &L;
};

/// \brief The catch-all function for logging, it can log any type not already
///        handled
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

/// \brief Specialization of writeToLog to emit a message
template<bool X>
inline void writeToLog(Logger<X> &This, const LogTerminator, int) {
  This.emit();
}

/// \brief A global registry for all the loggers
///
/// Loggers are usually global static variables in translation units, the role
/// of this class is collecting them.
class LoggersRegistry {
public:
  LoggersRegistry() {}

  void add(Logger<true> *L) { Loggers.push_back(L); }
  void add(Logger<false> *L) { (void) L; }

  void enable(llvm::StringRef Name) {
    for (Logger<true> *L : Loggers)
      if (L->name() == Name)
        L->enable();
  }

  void disable(llvm::StringRef Name) {
    for (Logger<true> *L : Loggers)
      if (L->name() == Name)
        L->disable();
  }

private:
  std::vector<Logger<true> *> Loggers;
};

extern llvm::ManagedStatic<LoggersRegistry> Loggers;

template<>
inline void Logger<true>::init() {
  Loggers->add(this);
}

template<>
inline void Logger<false>::init() {
}

extern template class Logger<true>;
extern template class Logger<false>;

class StreamWrapperBase {
public:
  virtual void flush(std::stringstream &Buffer) = 0;
  virtual ~StreamWrapperBase() {}
};

/// Class that wraps a stream of any type. It has a `flush` method, that, when
/// called with a `std::stringstream` copy all of its content in the wrapped
/// stream.
///
/// The main reason for this is being able to have a dumpInternal method in a
/// .cpp file while preserving a stream-agnostic dump method in the header.
template<typename O>
class StreamWrapper : public StreamWrapperBase {
public:
  StreamWrapper(O &Stream) : Stream(Stream) {}

  virtual void flush(std::stringstream &Buffer) override {
    Buffer.flush();
    Stream << Buffer.str();
    std::stringstream Empty;
    Buffer.swap(Empty);
  }

private:
  O &Stream;
};

#endif // DEBUG_H
