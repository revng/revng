#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <mutex>
#include <optional>
#include <thread>

#include "llvm/Support/Base64.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/ConstexprString.h"
#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/Common.h"
#include "revng/Support/Assert.h"

#include "Types.h"

inline constexpr auto TracingEnv = "REVNG_C_API_TRACE_PATH";
inline auto PointerStyle = llvm::HexPrintStyle::PrefixLower;

// The opposite of a std::recursive_mutex, if locked by the same thread it will
// assert (this is to avoid a deadlock/malformed output when tracing)
class OncePerThreadMutex {
private:
  std::optional<std::thread::id> ThreadId;
  std::mutex Mutex;

public:
  OncePerThreadMutex() {}

  void lock() {
    if (ThreadId.has_value())
      revng_assert(std::this_thread::get_id() != *ThreadId,
                   "NonRecursiveMutex entered twice by the same thread!");
    Mutex.lock();
    ThreadId = std::this_thread::get_id();
  }

  void unlock() {
    revng_assert(std::this_thread::get_id() == *ThreadId);
    Mutex.unlock();
    ThreadId.reset();
  }
};

inline OncePerThreadMutex TraceMutex;

// Helper class for tracing, this will be used by the the argument/return
// handlers defined below to properly print the value of the arguments onto the
// YAML tracing file
class TraceWriter {
private:
  llvm::raw_ostream &OS;
  // If true it means we're outputting a function's arguments, whereas if it is
  // false it means that we're outputting its return value.
  // This is needed because sometimes, for the same data type, we want to output
  // different things (e.g. Value vs Pointer) if it's an argument or a return
  // value
  bool OutputtingArguments = false;

  // Integer used to compute the ID of the command
  uint64_t ID = 0;

public:
  TraceWriter(llvm::raw_ostream &OS) : OS(OS) {
    OS.SetUnbuffered();
    printHeader();
  }

public:
  void functionPrelude(const llvm::StringRef Name) {
    OS << "- ID: " << ID++ << "\n";
    OS << "  StartTime: " << getUnixMillis() << "\n";
    OS << "  Name: " << Name << "\n";
    OS << "  Arguments:\n";
    OutputtingArguments = true;
    OS.flush();
  }

  void newArgument() {
    OS << "  - ";
    OS.flush();
  }

  // For integral types we still keep the template parameter. This is to avoid
  // the overload selector doing an implicit conversion of unexpected types to
  // these types
  template<IntegerType T>
  void printValue(const T &Int) {
    OS << Int << "\n";
    OS.flush();
  }

  template<typename T>
    requires std::is_same_v<T, bool>
  void printValue(const T &Bool) {
    OS << (Bool ? "true" : "false") << "\n";
    OS.flush();
  }

  template<typename T>
    requires std::is_same_v<T, char>
  void printValue(const T *String) {
    if (OutputtingArguments) {
      OS << reprString(String) << "\n";
      OS.flush();
    } else {
      printPointer(String);
    }
  }

  template<RPType T>
  void printValue(const T *Ptr) {
    printPointer(Ptr);
  }

  template<typename T>
  void printPointer(const T *Ptr) {
    // NOTE: if reading traces becomes a major task, it might be beneficial to
    // switch to a representation with increasing indexes, e.g. object_1,
    // object_2, ...
    OS << PointerPrefix;
    llvm::write_hex(OS, reinterpret_cast<uintptr_t>(Ptr), PointerStyle);
    OS << "\n";
    OS.flush();
  }

  void printBuffer(const llvm::StringRef Input) {
    OS << llvm::encodeBase64(Input) << "\n";
    OS.flush();
  }

  template<IntegerType T>
  void printList(const T IntList[], uint64_t Length) {
    using IntT = max_int<T>;
    OS << "[";
    for (uint64_t I = 0; I < Length; I++) {
      OS << static_cast<IntT>(IntList[I]);
      if (I < Length - 1) {
        OS << ", ";
      }
    }
    OS << "]\n";
    OS.flush();
  }

  template<typename T>
    requires std::is_same_v<T, char>
  void printList(const T *StringList[], uint64_t Length) {
    OS << "[";
    for (uint64_t I = 0; I < Length; I++) {
      OS << reprString(StringList[I]);
      if (I < Length - 1) {
        OS << ", ";
      }
    }
    OS << "]\n";
    OS.flush();
  }

  template<RPType T>
  void printList(const T *PtrList[], uint64_t Length) {
    OS << "[";
    for (uint64_t I = 0; I < Length; I++) {
      OS << PointerPrefix;
      llvm::write_hex(OS,
                      reinterpret_cast<uintptr_t>(PtrList[I]),
                      PointerStyle);
      if (I < Length - 1) {
        OS << ", ";
      }
    }
    OS << "]\n";
    OS.flush();
  }

  template<typename... T>
    requires(sizeof...(T) < 2)
  void printReturn(T... ReturnValue) {
    OutputtingArguments = false;
    OS << "  Result: ";
    if constexpr (sizeof...(T) == 0) {
      OS << "null\n";
    } else {
      printValue(ReturnValue...);
    }
    OS << "  EndTime: " << getUnixMillis() << "\n";
    OS.flush();
  }

private:
  void printHeader() {
    OS << "Version: 1\n";
    OS << "Commands:\n";
    OS.flush();
  }

  std::string reprString(const char *String) {
    return '"' + llvm::yaml::escape(String) + '"';
  }

  // Returns the number of milliseconds since epoch
  static uint64_t getUnixMillis() {
    namespace sc = std::chrono;
    auto Now = sc::system_clock::now().time_since_epoch();
    return sc::duration_cast<std::chrono::milliseconds>(Now).count();
  }
};

class TracingRuntime {
private:
  std::optional<TraceWriter> Writer;
  std::optional<llvm::raw_fd_ostream> OS;

public:
  TracingRuntime() {
    if (auto Path = llvm::sys::Process::GetEnv(TracingEnv)) {
      std::error_code EC;
      OS.emplace(*Path, EC);
      revng_assert(!EC);
      Writer.emplace(*OS);
    }
  }

  void swap(llvm::raw_ostream *NewOS = nullptr) {
    Writer.reset();
    OS.reset();
    if (NewOS != nullptr) {
      Writer.emplace(*NewOS);
    }
  }

  bool isEnabled() const { return Writer.has_value(); }

  TraceWriter &operator*() {
    revng_assert(Writer.has_value());
    return *Writer;
  }

  TraceWriter *operator->() {
    revng_assert(Writer.has_value());
    return &*Writer;
  }
};

inline TracingRuntime Tracing;

template<ConstexprString Name, int I, int N, typename... T>
inline void handleArgument(std::tuple<T...> Args) {
  Tracing->newArgument();
  using ArgT = decltype(std::get<I>(Args));
  using RArgT = std::remove_reference_t<ArgT>;
  ArgT Argument = std::get<I>(Args);
  constexpr int LengthHintValue = LengthHint<Name, I>;
  if constexpr (LengthHintValue >= 0) {
    using LengthT = decltype(std::get<LengthHintValue>(Args));
    static_assert(isInteger<std::remove_reference_t<LengthT>>());
    LengthT LengthArgument = std::get<LengthHintValue>(Args);
    // Handle arguments with length hints
    if constexpr (std::is_same_v<RArgT, const char *>) {
      // Buffer
      Tracing->printBuffer({ Argument, LengthArgument });
    } else {
      // Array-like
      Tracing->printList(Argument, LengthArgument);
    }
  } else {
    if constexpr (isDestroy<Name>()) {
      // _destroy methods always take 1 argument and it's always a pointer
      static_assert(N == 1);
      Tracing->printPointer(Argument);
    } else {
      Tracing->printValue(Argument);
    }
  }

  if constexpr (I + 1 < N)
    handleArgument<Name, I + 1, N>(Args);
}

template<ConstexprString Name, typename... T>
inline void handleArguments(T &&...Args) {
  if constexpr (sizeof...(T) > 0)
    handleArgument<Name, 0, sizeof...(T)>(std::make_tuple(Args...));
}

// This function will be used in each PipelineC function we need to wrap
// For example:
// rp_initialize(...) { return wrap<"rp_initialize">(_rp_initialize, ...); }
template<ConstexprString Name, typename CalleeT, typename... ArgsT>
inline decltype(auto) wrap(CalleeT Callee, ArgsT... Args) {
  using ReturnT = typename decltype(std::function{ Callee })::result_type;
  if (Tracing.isEnabled()) {
    // Special lock to avoid trace output being broken by multithreading or by
    // calling a PipelineC function within PipelineC
    std::lock_guard Guard(TraceMutex);

    Tracing->functionPrelude(std::string_view(Name));
    handleArguments<Name>(Args...);
    if constexpr (std::is_same_v<ReturnT, void>) {
      Callee(std::forward<ArgsT>(Args)...);
      Tracing->printReturn();
    } else {
      ReturnT Return = Callee(std::forward<ArgsT>(Args)...);
      Tracing->printReturn(Return);
      return Return;
    }
  } else {
    return Callee(std::forward<ArgsT>(Args)...);
  }
}
