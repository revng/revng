/// \file Runner.cpp
/// Implements the logic to run a trace file, this boils down to the Trace.run
/// function that will re-execute the commands of the trace in order.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>

#include "llvm/Support/Base64.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/ConstexprString.h"
#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/Common.h"
#include "revng/PipelineC/Tracing/Trace.h"
#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"

#include "Types.h"
#include "sanitizer/asan_interface.h"

using ArgumentRef = const revng::tracing::Argument &;
using ArgumentsRef = const llvm::ArrayRef<revng::tracing::Argument>;
using ReturnRef = const llvm::StringRef;

static Logger<> TraceRunnerLogger("trace-runner");

namespace utils {
// Utility functions for decoding arguments

// Utility class to store pointer arrays. When `.data()` is called, a nullptr is
// added at the end of the array and it's poisoned by ASAN. This allows to spot
// out of bounds accesses. After `.data()` has been called the internal vector
// cannot be modified unless `.clear()` is called.
template<typename T>
  requires std::is_pointer_v<T>
class PointerVector {
private:
  std::vector<T> Pointers;
  bool TailPoisoned = false;

public:
  PointerVector() {}
  ~PointerVector() { clear(); }

  PointerVector(const PointerVector &) = delete;
  PointerVector &operator=(const PointerVector &) = delete;
  PointerVector(PointerVector &&) = default;
  PointerVector &operator=(const PointerVector &&) = default;

  void push_back(T Pointer) {
    revng_assert(not TailPoisoned);
    Pointers.push_back(Pointer);
  }

  void clear() {
    if (TailPoisoned) {
      ASAN_UNPOISON_MEMORY_REGION(Pointers.back(), sizeof(T));
      TailPoisoned = false;
    }
    Pointers.clear();
  }

  T *data() {
    if (not TailPoisoned) {
      Pointers.push_back(nullptr);
      ASAN_POISON_MEMORY_REGION(Pointers.back(), sizeof(T));
      TailPoisoned = true;
    }
    return Pointers.data();
  }
};

// Utility class to represent a list of strings for C usage.
// It will copy the strings from the source string list and store them
// internally. It will also create an auxiliary list with pointers to the data
// to be used with a C-like API.
class CStringList {
private:
  std::vector<std::string> Storage;
  PointerVector<const char *> Pointers;

public:
  CStringList(const llvm::ArrayRef<llvm::StringRef> Source) {
    for (const llvm::StringRef &String : Source) {
      Storage.push_back(String.str());
    }
    updatePointers();
  }

  CStringList(std::vector<std::string> &&Source) : Storage(std::move(Source)) {
    updatePointers();
  }

public:
  const char **getCPointer() { return Pointers.data(); }

private:
  void updatePointers() {
    Pointers.clear();
    for (const std::string &String : Storage) {
      Pointers.push_back(String.c_str());
    }
  }
};

} // namespace utils

static class TemporaryDirectories {
private:
  inline static bool HandlerRegistered = false;
  inline static std::vector<std::string> Directories;

public:
  static std::string get(llvm::StringRef Prefix, bool Keep) {
    llvm::SmallString<128> Output;
    llvm::sys::fs::createUniqueDirectory(Prefix, Output);
    if (not Keep) {
      if (not HandlerRegistered) {
        llvm::sys::AddSignalHandler(doClean, nullptr);
        HandlerRegistered = true;
      }
      Directories.push_back(Output.str().str());
    }
    return Output.str().str();
  }

private:
  static void doClean(void *) {
    for (auto &Directory : Directories) {
      std::filesystem::remove_all(Directory);
    }
  }
} TemporaryDirectories;

class RunnerContext {
private:
  llvm::StringMap<uintptr_t> Pointers;
  bool ResumeDirectoryUsed = false;

public:
  const revng::tracing::RunTraceOptions Options;

public:
  RunnerContext(const revng::tracing::RunTraceOptions Options) :
    Options(Options) {
    if (!Options.TemporaryRoot.empty())
      revng_check(llvm::sys::fs::is_directory(Options.TemporaryRoot));
  }
  ~RunnerContext() = default;
  RunnerContext(const RunnerContext &Other) = delete;
  RunnerContext(RunnerContext &&Other) = delete;
  RunnerContext &operator=(const RunnerContext &Other) = delete;
  RunnerContext &operator=(RunnerContext &&Other) = delete;

  std::string getResumeDirectory() {
    if (not Options.ResumeDirectory.empty()) {
      revng_assert(not ResumeDirectoryUsed,
                   "ResumeDirectory specified with multiple managers");
      ResumeDirectoryUsed = true;
      return Options.ResumeDirectory;
    } else {
      return getTemporaryDirectory();
    }
  }

  std::string getTemporaryDirectory() {
    if (!Options.TemporaryRoot.empty())
      return TemporaryDirectories.get(Options.TemporaryRoot, true);
    else
      return TemporaryDirectories.get("revng-run-trace", false);
  }

  template<typename T>
  void storePointer(const llvm::StringRef Name, T *Pointer) {
    if (Name == NullPointer) {
      return;
    }
    Pointers[Name] = reinterpret_cast<uintptr_t>(Pointer);
  }

  template<typename T>
    requires std::is_pointer_v<T>
  T getPointer(const llvm::StringRef Name) {
    if (Name == NullPointer)
      return nullptr;

    revng_check(Pointers.count(Name) != 0,
                "Tried to retrieve a pointer from context that was not "
                "previously stored");
    return reinterpret_cast<T>(Pointers[Name]);
  }

  void invalidatePointer(const llvm::StringRef Name) { Pointers.erase(Name); }

private:
  template<ConstexprString Name, typename ArgT, size_t I>
    requires(anyOf<ArgT, char *, const char *>())
  std::vector<char> parseString(ArgumentRef Argument) {
    revng_check(Argument.isScalar(), "Argument is not scalar");
    constexpr int LH = LengthHint<Name, I>;
    std::vector<char> Result;
    if constexpr (LH > 0) {
      llvm::Error Err = llvm::decodeBase64(Argument.getScalar(), Result);
      revng_check(!Err);
    } else {
      Result.assign(Argument.getScalar().begin(), Argument.getScalar().end());
    }
    // Zero-terminate the string
    Result.push_back(0);
    return Result;
  }

  template<ConstexprString Name, IntegerType ArgT>
  ArgT parseArgumentImpl(ArgumentRef Argument) {
    return Argument.asInt<ArgT>();
  }

  template<ConstexprString Name, typename ArgT>
    requires(anyOf<ArgT, char **, const char **>())
  utils::CStringList parseArgumentImpl(ArgumentRef Argument) {
    return utils::CStringList(Argument.toStringList());
  }

  template<ConstexprString Name, typename ArgT>
    requires(std::is_pointer_v<ArgT>
             && isInteger<std::remove_pointer_t<ArgT>>())
  std::vector<std::remove_pointer_t<ArgT>>
  parseArgumentImpl(ArgumentRef Argument) {
    return Argument.toIntList<std::remove_pointer_t<ArgT>>();
  }

  template<ConstexprString Name, typename ArgT>
    requires(std::is_pointer_v<ArgT> && isRPType<remove_constptr<ArgT>>())
  ArgT parseArgumentImpl(ArgumentRef Argument) {
    revng_check(Argument.isScalar(), "Argument is not scalar");
    const std::string &PointerName = Argument.getScalar();
    ArgT Pointer = getPointer<ArgT>(PointerName);
    if constexpr (isDestroy<Name>())
      invalidatePointer(PointerName);
    return Pointer;
  }

  template<ConstexprString Name, typename ArgT>
    requires(isList<ArgT>()
             && isRPType<remove_constptr<remove_constptr<ArgT>>>())
  utils::PointerVector<std::remove_pointer_t<ArgT>>
  parseArgumentImpl(ArgumentRef Argument) {
    revng_check(Argument.isSequence(), "Argument is not a sequence");
    using ElementT = std::remove_pointer_t<ArgT>;
    utils::PointerVector<ElementT> Result;
    for (auto &Elem : Argument.getSequence()) {
      Result.push_back(getPointer<ElementT>(Elem));
      if constexpr (isDestroy<Name>())
        invalidatePointer(Elem);
    }

    return Result;
  }

public:
  template<ConstexprString Name, typename ArgT, size_t I>
  decltype(auto) parseArgument(ArgumentsRef Arguments) {
    if constexpr (anyOf<ArgT, char *, const char *>())
      return parseString<Name, ArgT, I>(Arguments[I]);
    else
      return parseArgumentImpl<Name, ArgT>(Arguments[I]);
  }

  template<ConstexprString Name, typename ArgT, size_t I, typename... Args>
  ArgT unwrapStorage(std::tuple<Args...> &Arguments) {
    if constexpr (isList<ArgT>()) {
      // In case of string lists the storage used is utils::CStringList
      if constexpr (anyOf<ArgT, char **, const char **>()) {
        return std::get<I>(Arguments).getCPointer();
      } else {
        return std::get<I>(Arguments).data();
      }
    } else if constexpr (anyOf<ArgT, char *, const char *>()) {
      if constexpr (isDestroy<Name>()) {
        // _destroy method parameter is not a string, but a pointer to the
        // string's address
        std::string PointerName(std::get<I>(Arguments).data());
        ArgT Pointer = getPointer<ArgT>(PointerName);
        invalidatePointer(PointerName);
        return Pointer;
      } else {
        // Strings are stored as std::vector<char>
        return std::get<I>(Arguments).data();
      }
    } else {
      return std::get<I>(Arguments);
    }
  }
};

using CommandRunner = std::function<void(/* format */ RunnerContext &Context,
                                         ArgumentsRef Arguments,
                                         ReturnRef Return)>;

static void
softAssert(const RunnerContext &Context, bool Check, const char *Message) {
  if (Context.Options.SoftAsserts) {
    if (!Check) {
      TraceRunnerLogger << "Warning: Assertion failed: " << Message << "\n";
    }
  } else {
    revng_check(Check, Message);
  }
}

template<ConstexprString Name, typename ReturnT, typename... Args, size_t... I>
static ReturnT runCommand(std::function<ReturnT(Args...)> Function,
                          ArgumentsRef Arguments,
                          RunnerContext &Context,
                          std::index_sequence<I...> Seq) {
  using std::make_tuple;
  auto &Arg = Arguments;
  auto Storage = make_tuple(Context.parseArgument<Name, Args, I>(Arg)...);
  return Function(Context.unwrapStorage<Name, Args, I>(Storage)...);
}

template<ConstexprString Name, typename ReturnT, typename... Args>
static void runnerImplementation(std::function<ReturnT(Args...)> Function,
                                 RunnerContext &Context,
                                 ArgumentsRef Arguments,
                                 ReturnRef Return) {
  revng_check(sizeof...(Args) == Arguments.size(),
              "Mismatch between the number of arguments in the trace and the "
              "function's prototype");

  auto Sequence = std::make_index_sequence<sizeof...(Args)>();
  if constexpr (is_same_v<ReturnT, void>) {
    runCommand<Name>(Function, Arguments, Context, Sequence);
    return;
  } else {
    ReturnT Ret = runCommand<Name>(Function, Arguments, Context, Sequence);

    // Check return value for integral return types
    if constexpr (anyOf<ReturnT, uint8_t, uint32_t, uint64_t>()) {
      // Artifact generation is not stable, rp_buffer_size will, for sure,
      // report a different size on different runs
      if constexpr (std::string_view(Name) != "rp_buffer_size") {
        softAssert(Context,
                   detail::toInt<ReturnT>(Return) == Ret,
                   "Return value differs from what happened in the trace");
      }
    } else if constexpr (std::is_same_v<ReturnT, bool>) {
      softAssert(Context,
                 Ret == *llvm::yaml::parseBool(Return),
                 "Return value differs from what happened in the trace");
    }

    if (Return.startswith(PointerPrefix)) {
      if constexpr (std::is_pointer_v<ReturnT>) {
        // Check that pointers are NULL or non-NULL
        if (Return == NullPointer) {
          revng_check(reinterpret_cast<uintptr_t>(Ret) == 0,
                      "Function that returned a null pointer in the trace "
                      "returned a non-null pointer");
        } else {
          softAssert(Context,
                     reinterpret_cast<uintptr_t>(Ret) != 0,
                     "Function that returned a non-null pointer in the trace "
                     "returned a null pointer");
        }

        // Store the returned pointer
        Context.storePointer(Return, Ret);
      } else {
        revng_abort();
      }
    }
  }
}

template<ConstexprString Name, typename ReturnT, typename... Args>
static CommandRunner makeRunner(std::function<ReturnT(Args...)> Function) {
  return [Function](RunnerContext &Context,
                    ArgumentsRef Arguments,
                    ReturnRef Return) {
    runnerImplementation<Name>(Function, Context, Arguments, Return);
  };
}

// rp_initialize causes an abort, this is done since there can only be one call
// to it for the entire duration of the program and it is assumed that the
// caller of Trace::run will have done so beforehand. We still allow traces to
// have the first command be an rp_initialize for ease of use.
static void handleRpInitialize(RunnerContext &Context,
                               ArgumentsRef Arguments,
                               ReturnRef Return) {
  softAssert(Context, false, "rp_initialize invoked mid-trace");
}

// Ditto as above, but for the last instruction
static void handleRpShutdown(RunnerContext &Context,
                             ArgumentsRef Arguments,
                             ReturnRef Return) {
  softAssert(Context, false, "rp_shutdown invoked mid-trace");
}

class CommandHandler {
private:
  llvm::StringMap<CommandRunner> Registry;

public:
  CommandHandler() {
    Registry.insert({ "rp_initialize", handleRpInitialize });
    Registry.insert({ "rp_shutdown", handleRpShutdown });

// Add autogenerated registerRunner calls to register the
// other functions automatically
#define FUNCTION(fname) registerFunction<#fname>(fname);
#include "revng/PipelineC/Functions.inc"
#undef FUNCTION
  }

public:
  bool has(const llvm::StringRef Name) const {
    return Registry.count(Name) == 1;
  }

  CommandRunner &operator[](const llvm::StringRef Name) {
    return Registry[Name];
  }

private:
  template<ConstexprString Name, typename FunctionT>
  void registerFunction(FunctionT Function) {
    std::string StrName{ llvm::StringRef(Name) };
    if (Registry.count(StrName) == 0) {
      CommandRunner Runner = makeRunner<Name>(std::function{ Function });
      Registry.insert({ StrName, Runner });
    }
  }
};

static std::vector<revng::tracing::Argument>
argumentTransformer(RunnerContext &Context,
                    const revng::tracing::Command &Command) {
  decltype(Command.Arguments) NewArguments(Command.Arguments);
  auto ReplaceWithResumeDirectory = [&](size_t Index) {
    revng_assert(Command.Arguments[Index].isScalar(), "Argument is not scalar");
    NewArguments[Index].getScalar() = Context.getResumeDirectory();
  };

  // Replace workdir with a temporary directory
  if (Command.Name == "rp_manager_create") {
    ReplaceWithResumeDirectory(2);
  } else if (Command.Name == "rp_manager_create_from_string") {
    ReplaceWithResumeDirectory(4);
  }

  return NewArguments;
}

namespace revng::tracing {
llvm::Error Trace::run(const revng::tracing::RunTraceOptions Options) const {
  using namespace revng;

  RunnerContext Context(Options);

  const tracing::Command &FirstCommand = this->Commands.front();
  const tracing::Command &LastCommand = this->Commands.back();

  // Allow rp_initialize as first command and rp_shutdown as last, in all other
  // cases the trace is malformed and needs to be aborted
  const size_t FirstCommandI = FirstCommand.Name == "rp_initialize" ? 1 : 0;
  const size_t LastCommandI = this->Commands.size()
                              - (LastCommand.Name == "rp_shutdown" ? 1 : 0);

  static CommandHandler CommandHandler;
  for (size_t CommandI = FirstCommandI; CommandI < LastCommandI; CommandI++) {
    auto &Command = this->Commands[CommandI];
    revng_check(CommandHandler.has(Command.Name),
                "Command handler for command not found");
    auto Arguments = argumentTransformer(Context, Command);

    if (Options.BreakAt.contains(CommandI))
      raise(SIGTRAP);

    CommandHandler[Command.Name](Context, Arguments, Command.Result);
  }

  return llvm::Error::success();
};
} // namespace revng::tracing
