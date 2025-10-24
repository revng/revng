#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Use.h"

#include "revng/Support/IRHelpers.h"

#include "Context.h"
#include "Value.h"

inline Logger<> Log("argument-usage-analysis");

template<typename O>
void dumpCall(O &Output, const llvm::CallInst *Call) {
  const llvm::Function *Callee = getCalledFunction(Call);
  Output << "Call " << getName(Call) << " in "
         << Call->getParent()->getParent()->getName().str();
  if (Callee == nullptr) {
    Output << " (indirect)";
  } else {
    Output << " targeting " << Callee->getName().str();
  }
}

namespace aua {

class MemoryAccess {
private:
  const llvm::Use *Location = nullptr;
  const Value *Start = nullptr;

public:
  MemoryAccess(const llvm::Use &Location, const Value &Start) :
    Location(&Location), Start(&Start) {}

public:
  MemoryAccess replaceStart(const Value &NewStart) const {
    MemoryAccess Result = *this;
    Result.Start = &NewStart;
    return Result;
  }

public:
  std::strong_ordering operator<=>(const MemoryAccess &Other) const = default;

public:
  [[nodiscard]] const llvm::Use &location() const { return *Location; }
  [[nodiscard]] const Value &start() const { return *Start; }
  [[nodiscard]] bool isWrite() const { return isWrite(Location); }

public:
  [[nodiscard]] static bool isWrite(const llvm::Use *Use) {
    llvm::User *U = Use->getUser();
    if (isa<llvm::StoreInst>(U)) {
      return true;
    } else if (isa<llvm::LoadInst>(U)) {
      return false;
    } else if (auto *Call = dyn_cast<llvm::CallInst>(U)) {
      return Use->getOperandNo() == 0;
    }

    revng_abort();
  }

public:
  std::string toString() const {
    return Start->toString() + " (operand "
           + std::to_string(Location->getOperandNo()) + " of "
           + getName(Location->getUser()) + ")";
  }
};

class EscapedArgument {
private:
  unsigned ArgumentIndex = 0;
  llvm::Instruction *Location = nullptr;

public:
  EscapedArgument(llvm::Instruction &Location, unsigned ArgumentIndex) :
    ArgumentIndex(ArgumentIndex), Location(&Location) {}

public:
  std::strong_ordering
  operator<=>(const EscapedArgument &Other) const = default;

public:
  unsigned index() const { return ArgumentIndex; }
  llvm::Instruction &location() const { return *Location; }

public:
  std::string toString() const { return "Argument" + std::to_string(index()); }
};

class Call {
  friend std::hash<Call>;

private:
  /// The call is here for debugging purposes only
  const llvm::CallInst *CallInstruction = nullptr;
  const Function *Callee = nullptr;
  llvm::DenseMap<uint64_t, const Value *> ActualArguments;

public:
  Call(const llvm::CallInst &CallInstruction, const Function &Callee) :
    CallInstruction(&CallInstruction), Callee(&Callee) {}

  bool operator<(const Call &Other) const {
    auto ToSortable =
      [](const llvm::DenseMap<uint64_t, const Value *> &ToSort) {
        SmallVector<std::pair<uint64_t, const Value *>> Result;
        for (auto &&[Index, Value] : ToSort)
          Result.push_back({ Index, Value });
        llvm::sort(Result);
        return Result;
      };
    auto ThisSortableActualArguments = ToSortable(ActualArguments);
    auto ThisTuple = std::tie(CallInstruction,
                              Callee,
                              ThisSortableActualArguments);

    auto OtherSortableActualArguments = ToSortable(Other.ActualArguments);
    auto OtherTuple = std::tie(Other.CallInstruction,
                               Other.Callee,
                               OtherSortableActualArguments);
    return ThisTuple < OtherTuple;
  }

  bool operator==(const Call &Other) const = default;

public:
  const llvm::CallInst &callInstruction() const { return *CallInstruction; }
  const Function &callee() const { return *Callee; }
  auto &actualArguments() { return ActualArguments; }
  const llvm::DenseMap<uint64_t, const Value *> &actualArguments() const {
    return ActualArguments;
  }

public:
  void registerActualArgument(uint64_t Index, const Value &Argument) {
    revng_assert(ActualArguments.count(Index) == 0);
    ActualArguments[Index] = &Argument;
  }

public:
  template<typename O>
  void dump(O &Output, llvm::StringRef Prefix, bool JustArguments) const {
    std::string Indent;
    if (not JustArguments) {
      Output << Prefix.str();
      dumpCall(Output, CallInstruction);
      Output << "\n";
      Indent = "  ";
    }

    Output << Prefix.str() << Indent << "Actual arguments:\n";
    for (uint64_t Index = 0; Index < ActualArguments.size(); ++Index) {
      Output << Prefix.str() << Indent << "  " << Index << ". "
             << ActualArguments.find(Index)->second->toString() << "\n";
    }
  }

  void dump() const debug_function { dump(dbg, "", false); }
};

class Function {
private:
  Context &TheContext;
  std::set<MemoryAccess> LocalAccesses;
  std::set<EscapedArgument> LocalEscapedArguments;
  SmallVector<Call, 2> Calls;
  const Value *ReturnValue = nullptr;
  llvm::DenseMap<const llvm::Value *, const Value *> AnalysisResult;

public:
  Function(Context &TheContext) : TheContext(TheContext) {}

public:
  const Value *tryGet(const llvm::Value &V) const {
    auto It = AnalysisResult.find(&V);
    if (It == AnalysisResult.end()) {
      if (auto *Constant = dyn_cast<llvm::ConstantInt>(&V)) {
        if (Constant->getBitWidth() <= 64) {
          // Handle constants on the fly
          return &TheContext.getConstant(getLimitedValue(Constant));
        } else {
          // Ignore the rest
          return &TheContext.getUnknown();
        }
      } else if (isa<llvm::Constant>(&V)) {
        // Ignore globals, function pointer, constant expressions and so on
        return &TheContext.getUnknown();
      } else {
        return nullptr;
      }
    } else {
      return It->second;
    }
  }

  const Value &get(const llvm::Value &V) const {
    const Value *Result = tryGet(V);
    revng_assert(Result != nullptr, dumpToString(V).c_str());
    return *Result;
  }

  const auto &localAccesses() const { return LocalAccesses; }
  const SmallVector<Call, 2> &calls() const { return Calls; }
  const Value *returnValue() const { return ReturnValue; }
  const auto &localEscapedArguments() const { return LocalEscapedArguments; }

public:
  void set(const llvm::Value &V, const Value &NewValue) {
    AnalysisResult[&V] = &NewValue;
  }

  void registerAccess(const llvm::Use &Location, const Value &NewValue) {
    LocalAccesses.insert({ Location, NewValue });
  }

  void logEscapedValue(llvm::StringRef Context, const Value &EscapedValue) {
    if (Log.isEnabled()) {
      llvm::DenseSet<unsigned> EscapedArguments = EscapedValue
                                                    .collectArguments();
      if (EscapedArguments.size() != 0) {
        Log << "In " << Context.str()
            << " the following value escapes: " << EscapedValue.toString()
            << ". Therefore, the following arguments escape: {";
        for (unsigned ArgumentIndex : EscapedArguments)
          Log << " " << ArgumentIndex;
        Log << " }" << DoLog;
      }
    }
  }

  void registerEscapedValue(llvm::Instruction &Location,
                            const Value &EscapedValue) {
    for (unsigned ArgumentIndex : EscapedValue.collectArguments())
      LocalEscapedArguments.insert({ Location, ArgumentIndex });
  }

  void registerReturnValue(Context &Context, const Value &NewReturnValue) {
    if (ReturnValue == nullptr) {
      ReturnValue = &NewReturnValue;
    } else {
      ReturnValue = &Context.getAnyOf({ ReturnValue, &NewReturnValue });
    }
  }

  void registerCall(aua::Call &&NewCall) {
    Calls.push_back(std::move(NewCall));
  }

public:
  template<typename O>
  void dump(O &Output, llvm::StringRef Prefix) const {
    if (LocalAccesses.size() > 0) {
      Output << Prefix.str() << "Local reads:\n";
      std::set<const Value *> Reads;
      for (const MemoryAccess &Access : LocalAccesses)
        if (not Access.isWrite())
          Reads.insert(&Access.start());

      for (const Value *Read : Reads)
        Output << Prefix.str() << "  " << Read->toString() << "\n";

      Output << Prefix.str() << "Local writes:\n";
      std::set<const Value *> Writes;
      for (const MemoryAccess &Access : LocalAccesses)
        if (Access.isWrite())
          Writes.insert(&Access.start());

      for (const Value *Write : Writes)
        Output << Prefix.str() << "  " << Write->toString() << "\n";
    }

    if (LocalEscapedArguments.size() > 0) {
      Output << Prefix.str() << "Local escaped arguments: {";
      for (const EscapedArgument &EscapedArgument : LocalEscapedArguments)
        Output << " " << EscapedArgument.toString();
      Output << " }\n";
    }

    Output << Prefix.str() << "Return value: " << returnValue()->toString()
           << "\n";
  }

  void dump() const debug_function { dump(dbg, ""); }
};

} // namespace aua

// Specialization of std::hash for Lel
namespace std {
template<>
struct hash<aua::Call> {
  std::size_t operator()(const aua::Call &Call) const {
    using namespace llvm;
    std::size_t Hash = 0;
    Hash = hash_combine(Hash,
                        reinterpret_cast<std::uintptr_t>(Call.CallInstruction));
    Hash = hash_combine(Hash, reinterpret_cast<std::uintptr_t>(Call.Callee));
    for (const auto &[Index, Argument] : Call.ActualArguments) {
      Hash = hash_combine(Hash, Index);
      Hash = hash_combine(Hash, reinterpret_cast<std::uintptr_t>(Argument));
    }
    return Hash;
  }
};
} // namespace std
