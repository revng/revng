#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"

namespace detail {

template<typename T>
inline T toInt(const llvm::StringRef StrInt) {
  T Result;
  revng_assert(!StrInt.getAsInteger(10, Result));
  return Result;
}

} // namespace detail

namespace revng::tracing {

enum ArgumentState {
  Invalid,
  Scalar,
  Sequence
};

struct Argument {
private:
  ArgumentState State = Invalid;
  std::string Scalar;
  std::vector<std::string> Sequence;

public:
  bool isValid() const { return State != ArgumentState::Invalid; }
  bool isScalar() const { return State == ArgumentState::Scalar; }
  bool isSequence() const { return State == ArgumentState::Sequence; }

  std::string &getScalar() {
    setState(ArgumentState::Scalar);
    return Scalar;
  }

  std::vector<std::string> &getSequence() {
    setState(ArgumentState::Sequence);
    return Sequence;
  }

  const std::string &getScalar() const {
    revng_assert(State == ArgumentState::Scalar);
    return Scalar;
  }

  const std::vector<std::string> &getSequence() const {
    revng_assert(State == ArgumentState::Sequence);
    return Sequence;
  }

public:
  template<typename T>
  T asInt() const {
    revng_assert(isScalar());
    return ::detail::toInt<T>(Scalar);
  }

  template<typename T>
  std::vector<T> toIntList() const {
    revng_assert(isSequence());
    std::vector<T> Result;
    for (auto &Elem : Sequence) {
      Result.push_back(::detail::toInt<T>(Elem));
    }
    return Result;
  }

  std::vector<std::string> toStringList() const {
    revng_assert(isSequence());
    return Sequence;
  }

private:
  void setState(const ArgumentState NewState) {
    revng_assert(State == Invalid || State == NewState);
    State = NewState;
  }
};

struct Command {
public:
  uint64_t ID;
  uint64_t StartTime;
  std::string Name;
  std::vector<Argument> Arguments;
  std::string Result;
  uint64_t EndTime;

public:
  void dump(llvm::raw_ostream &Stream) const;

  void dump() const debug_function {
    llvm::raw_os_ostream OS(dbg);
    dump(OS);
  }
};

struct BufferLocation {
  std::string CommandName;
  size_t CommandNumber;
  size_t ArgumentNumber;
};

struct RunTraceOptions {
public:
  // If true some assertions will result in a warning rather than aborting
  bool SoftAsserts = false;
  // List of Command Indexes to break at when debugging
  std::set<uint64_t> BreakAt;
  // If set, all temporary directories will be created within this directory
  std::string TemporaryRoot;
  // Instead of using a temporary directory, the first invocation will use
  // these directory instead and subsequent ones will abort
  std::string ResumeDirectory;
};

struct Trace {
public:
  uint64_t Version;
  std::vector<Command> Commands;

public:
  std::vector<BufferLocation> listBuffers() const;
  llvm::Expected<std::vector<char>>
  getBuffer(const BufferLocation &Location) const;
  llvm::Expected<std::vector<char>> getBuffer(size_t CommandNo,
                                              size_t ArgNo) const;
  llvm::Error run(const RunTraceOptions Options = {}) const;

public:
  static llvm::Expected<Trace> fromFile(const llvm::StringRef Path);
  static llvm::Expected<Trace> fromBuffer(const llvm::MemoryBuffer &Buffer);
};

} // namespace revng::tracing

template<>
struct llvm::yaml::MappingTraits<revng::tracing::Trace> {
  static void mapping(IO &TheIO, revng::tracing::Trace &Trace) {
    TheIO.mapRequired("Version", Trace.Version);
    TheIO.mapRequired("Commands", Trace.Commands);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(revng::tracing::Argument);
LLVM_YAML_IS_SEQUENCE_VECTOR(revng::tracing::Command);

template<>
struct llvm::yaml::MappingTraits<revng::tracing::Command> {
  static void mapping(IO &TheIO, revng::tracing::Command &TraceCommand) {
    TheIO.mapRequired("ID", TraceCommand.ID);
    TheIO.mapRequired("Name", TraceCommand.Name);
    TheIO.mapRequired("StartTime", TraceCommand.StartTime);
    TheIO.mapRequired("Arguments", TraceCommand.Arguments);
    TheIO.mapOptional("Result", TraceCommand.Result);
    TheIO.mapOptional("EndTime", TraceCommand.EndTime);
  }
};

template<>
struct llvm::yaml::PolymorphicTraits<revng::tracing::Argument> {
  static llvm::yaml::NodeKind
  getKind(const revng::tracing::Argument &Argument) {
    using llvm::yaml::NodeKind;
    return Argument.isScalar() ? NodeKind::Scalar : NodeKind::Sequence;
  }

  static std::string &getAsScalar(revng::tracing::Argument &Argument) {
    return Argument.getScalar();
  }

  static std::vector<std::string> &
  getAsSequence(revng::tracing::Argument &Argument) {
    return Argument.getSequence();
  }

  static revng::tracing::Argument &
  getAsMap(revng::tracing::Argument &Argument) {
    revng_abort();
  }
};

namespace revng::tracing {

inline llvm::Expected<Trace> Trace::fromFile(const llvm::StringRef Path) {
  auto MaybeInputBuffer = llvm::MemoryBuffer::getFileAsStream(Path);
  if (std::error_code EC = MaybeInputBuffer.getError()) {
    return llvm::createStringError(EC,
                                   "Unable to read input trace: "
                                     + EC.message());
  }

  return Trace::fromBuffer(**MaybeInputBuffer);
}

inline llvm::Expected<Trace>
Trace::fromBuffer(const llvm::MemoryBuffer &Buffer) {
  llvm::yaml::Input YAMLReader(Buffer);
  Trace Trace;
  YAMLReader >> Trace;

  if (Trace.Version != 1) {
    return revng::createError("Unexpected trace version: %u", Trace.Version);
  }

  for (size_t CommandI = 0; CommandI < Trace.Commands.size(); CommandI++) {
    auto &Command = Trace.Commands[CommandI];
    for (size_t ArgumentI = 0; ArgumentI < Command.Arguments.size();
         ArgumentI++) {
      auto &Argument = Command.Arguments[ArgumentI];
      if (!Argument.isValid())
        return revng::createError("Argument did not verify: Command #%u, "
                                  "Argument #%u",
                                  CommandI,
                                  ArgumentI);
    }
  }

  return Trace;
}

inline void Command::dump(llvm::raw_ostream &Stream) const {
  Stream << "\n";
  llvm::yaml::Output YAMLOutput(Stream);
  YAMLOutput << *const_cast<Command *>(this);
  Stream << "\n";
  Stream.flush();
}

} // namespace revng::tracing
