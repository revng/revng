#ifndef FUNCTIONSSUMMARY_H
#define FUNCTIONSSUMMARY_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file is NOT automatically generated.

// Standard includes
#include <deque>
#include <map>
#include <set>

// Local libraries includes
#include "revng/Support/Debug.h"

namespace llvm {
class BasicBlock;
class GlobalVariable;
class Instruction;
class Module;
class Value;
} // namespace llvm

namespace StackAnalysis {

namespace FunctionType {

enum Values {
  Invalid, ///< An invalid entry
  Regular, ///< A normal function
  NoReturn, ///< A noreturn function
  Fake ///< A fake function
};

inline const char *getName(Values Type) {
  switch (Type) {
  case Invalid:
    return "Invalid";
  case Regular:
    return "Regular";
  case NoReturn:
    return "NoReturn";
  case Fake:
    return "Fake";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid")
    return Invalid;
  else if (Name == "Regular")
    return Regular;
  else if (Name == "NoReturn")
    return NoReturn;
  else if (Name == "Fake")
    return Fake;
  else
    revng_abort();
}

} // namespace FunctionType

/// \brief Intraprocedural analysis interruption reasons
namespace BranchType {

enum Values {
  /// Invalid value
  Invalid,
  /// Branch due to instruction-level CFG (e.g., conditional move)
  InstructionLocalCFG,
  /// Branch due to function-local CFG (a regular branch)
  FunctionLocalCFG,
  /// A call to a fake function
  FakeFunctionCall,
  /// A return from a fake function
  FakeFunctionReturn,
  /// A function call for which the cache was able to produce a summary
  HandledCall,
  /// A function call for which the target is unknown
  IndirectCall,
  /// A function call for which the cache was not able to produce a summary
  UnhandledCall,
  /// A proper function return
  Return,
  /// A branch returning to the return address, but leaving the stack
  /// in an unexpected situation
  BrokenReturn,
  /// A branch representing an indirect tail call
  IndirectTailCall,
  /// A branch representing a longjmp or similar constructs
  LongJmp,
  /// A killer basic block (killer syscall or endless loop)
  Killer,
  /// The basic block ends with an unreachable instruction
  Unreachable,

  /// This function is fake, inform the interprocedural part of the analysis
  FakeFunction,
  /// The analysis of the function is finished and a summary is available
  RegularFunction,
  /// This is a function for which we couldn't find any return statement
  NoReturnFunction,
};

inline const char *getName(Values Type) {
  switch (Type) {
  case Invalid:
    return "Invalid";
  case InstructionLocalCFG:
    return "InstructionLocalCFG";
  case FunctionLocalCFG:
    return "FunctionLocalCFG";
  case FakeFunctionCall:
    return "FakeFunctionCall";
  case FakeFunctionReturn:
    return "FakeFunctionReturn";
  case HandledCall:
    return "HandledCall";
  case IndirectCall:
    return "IndirectCall";
  case UnhandledCall:
    return "UnhandledCall";
  case Return:
    return "Return";
  case BrokenReturn:
    return "BrokenReturn";
  case IndirectTailCall:
    return "IndirectTailCall";
  case FakeFunction:
    return "FakeFunction";
  case LongJmp:
    return "LongJmp";
  case Killer:
    return "Killer";
  case RegularFunction:
    return "RegularFunction";
  case NoReturnFunction:
    return "NoReturnFunction";
  case Unreachable:
    return "Unreachable";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid")
    return Invalid;
  else if (Name == "InstructionLocalCFG")
    return InstructionLocalCFG;
  else if (Name == "FunctionLocalCFG")
    return FunctionLocalCFG;
  else if (Name == "FakeFunctionCall")
    return FakeFunctionCall;
  else if (Name == "FakeFunctionReturn")
    return FakeFunctionReturn;
  else if (Name == "HandledCall")
    return HandledCall;
  else if (Name == "IndirectCall")
    return IndirectCall;
  else if (Name == "UnhandledCall")
    return UnhandledCall;
  else if (Name == "Return")
    return Return;
  else if (Name == "BrokenReturn")
    return BrokenReturn;
  else if (Name == "IndirectTailCall")
    return IndirectTailCall;
  else if (Name == "FakeFunction")
    return FakeFunction;
  else if (Name == "LongJmp")
    return LongJmp;
  else if (Name == "Killer")
    return Killer;
  else if (Name == "RegularFunction")
    return RegularFunction;
  else if (Name == "NoReturnFunction")
    return NoReturnFunction;
  else if (Name == "Unreachable")
    return Unreachable;
  else
    revng_abort();
}

} // namespace BranchType

/// \brief Class representing the state of a register in terms of being an
///        argument of a function or a function call
///
/// Collects information from URAOF, DRAOF and RAOFC.
///
/// \tparam FunctionCall true if this class represents the state of a register
///         in terms of being an argument of a function call (as opposed to a
///         function).
template<bool FunctionCall>
class RegisterArgument {
  // We're friends with the class with opposite FunctionCall status
  friend class RegisterArgument<not FunctionCall>;

  friend struct CombineHelper;

public:
  enum Values { No, NoOrDead, Dead, Yes, Maybe, Contradiction };

private:
  Values Value;

public:
  RegisterArgument() : Value(Maybe) {}

  RegisterArgument(Values Value) : Value(Value) {}

  static RegisterArgument no() {
    RegisterArgument Result;
    Result.Value = No;
    return Result;
  }

  static RegisterArgument maybe() {
    RegisterArgument Result;
    Result.Value = Maybe;
    return Result;
  }

  static RegisterArgument fromName(llvm::StringRef Name) {
    RegisterArgument Result;
    if (Name == "No")
      Result.Value = No;
    else if (Name == "NoOrDead")
      Result.Value = NoOrDead;
    else if (Name == "Dead")
      Result.Value = Dead;
    else if (Name == "Yes")
      Result.Value = Yes;
    else if (Name == "Maybe")
      Result.Value = Maybe;
    else if (Name == "Contradiction")
      Result.Value = Contradiction;
    else
      revng_abort();

    return Result;
  }

public:
  bool isContradiction() const { return Value == Contradiction; }

  /// This RegisterArgument concerns a function call for which either the callee
  /// is unknown or doesn't use this register at all.
  void notAvailable() {
    revng_assert(FunctionCall);

    // All the situations which embed the possibility of being an argument have
    // to go to a case that includes the possibility of *not* being an argument
    // due to the register being a callee-saved register
    switch (Value) {
    case Maybe:
      // These are already good
      break;
    case Yes:
      // Weaken the statement
      Value = Maybe;
      break;
    case No:
    case NoOrDead:
    case Contradiction:
    case Dead:
      revng_abort();
    }
  }

  void combine(const RegisterArgument<not FunctionCall> &Other);

  bool isCompatibleWith(const RegisterArgument<false> &Other) const {
    switch (Value) {
    case NoOrDead:
      switch (Other.Value) {
      case RegisterArgument<false>::NoOrDead:
      case RegisterArgument<false>::Dead:
      case RegisterArgument<false>::No:
      case RegisterArgument<false>::Maybe:
        return true;
      case RegisterArgument<false>::Yes:
      case RegisterArgument<false>::Contradiction:
        return false;
      }
      break;
    case Maybe:
      switch (Other.Value) {
      case RegisterArgument<false>::Maybe:
      case RegisterArgument<false>::NoOrDead:
      case RegisterArgument<false>::Dead:
      case RegisterArgument<false>::No:
      case RegisterArgument<false>::Yes:
        return true;
      case RegisterArgument<false>::Contradiction:
        return false;
      }
      break;
    case Yes:
      switch (Other.Value) {
      case RegisterArgument<false>::Yes:
      case RegisterArgument<false>::Maybe:
        return true;
      case RegisterArgument<false>::NoOrDead:
      case RegisterArgument<false>::Dead:
      case RegisterArgument<false>::No:
      case RegisterArgument<false>::Contradiction:
        return false;
      }
      break;
    case Dead:
      switch (Other.Value) {
      case RegisterArgument<false>::Dead:
      case RegisterArgument<false>::NoOrDead:
      case RegisterArgument<false>::Maybe:
        return true;
      case RegisterArgument<false>::Yes:
      case RegisterArgument<false>::No:
      case RegisterArgument<false>::Contradiction:
        return false;
      }
      break;
    case Contradiction:
      return false;
    case No:
      switch (Other.Value) {
      case RegisterArgument<false>::No:
      case RegisterArgument<false>::NoOrDead:
      case RegisterArgument<false>::Maybe:
        return true;
      case RegisterArgument<false>::Dead:
      case RegisterArgument<false>::Yes:
      case RegisterArgument<false>::Contradiction:
        return false;
      }
      break;
    }

    revng_abort();
  }

  const char *valueName() const {
    switch (Value) {
    case NoOrDead:
      return "NoOrDead";
    case Maybe:
      return "Maybe";
    case Yes:
      return "Yes";
    case Dead:
      return "Dead";
    case Contradiction:
      return "Contradiction";
    case No:
      return "No";
    }

    revng_abort();
  }

  Values value() const { return Value; }

  void dump() const { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << valueName();
  }
};

// Assign a name to the two possible options
using FunctionRegisterArgument = RegisterArgument<false>;
using FunctionCallRegisterArgument = RegisterArgument<true>;

template<>
void RegisterArgument<true>::combine(const RegisterArgument<false> &Other);
template<>
void RegisterArgument<false>::combine(const RegisterArgument<true> &Other);

// Let includers know that someone will define the two classes
extern template class RegisterArgument<true>;
extern template class RegisterArgument<false>;

class FunctionCallReturnValue;

/// \brief Class representing the state of a register in terms of being the
///        return value of a function.
///
/// Collects information from URVOF.
class FunctionReturnValue {
  friend class FunctionCallReturnValue;

  friend struct CombineHelper;

public:
  enum Values { No, NoOrDead, Maybe, Contradiction, YesOrDead };

private:
  Values Value;

public:
  FunctionReturnValue() : Value(Maybe) {}

  FunctionReturnValue(Values Value) : Value(Value) {}

  static FunctionReturnValue no() {
    FunctionReturnValue Result;
    Result.Value = No;
    return Result;
  }

  static FunctionReturnValue maybe() {
    FunctionReturnValue Result;
    Result.Value = Maybe;
    return Result;
  }

  static FunctionReturnValue fromName(llvm::StringRef Name) {
    FunctionReturnValue Result;
    if (Name == "No")
      Result.Value = No;
    else if (Name == "NoOrDead")
      Result.Value = NoOrDead;
    else if (Name == "Maybe")
      Result.Value = Maybe;
    else if (Name == "Contradiction")
      Result.Value = Contradiction;
    else if (Name == "YesOrDead")
      Result.Value = YesOrDead;
    else
      revng_abort();
    return Result;
  }

public:
  bool isContradiction() const { return Value == Contradiction; }

  void notAvailable() { revng_abort(); }

  void combine(const FunctionCallReturnValue &Other);

  const char *valueName() const {
    switch (Value) {
    case NoOrDead:
      return "NoOrDead";
    case Maybe:
      return "Maybe";
    case No:
      return "No";
    case Contradiction:
      return "Contradiction";
    case YesOrDead:
      return "YesOrDead";
    }

    revng_abort();
  }

  Values value() const { return Value; }

  void dump() const { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << valueName();
  }
};

/// \brief Class representing the state of a register in terms of being the
///        return value of a function call.
///
/// Collects information from DRVOFC and URVOFC.
class FunctionCallReturnValue {
  friend class FunctionReturnValue;

  friend struct CombineHelper;

public:
  enum Values { No, NoOrDead, Maybe, Yes, Dead, Contradiction, YesOrDead };

private:
  Values Value;

public:
  FunctionCallReturnValue() : Value(Maybe) {}

  FunctionCallReturnValue(Values Value) : Value(Value) {}

  static FunctionCallReturnValue no() {
    FunctionCallReturnValue Result;
    Result.Value = No;
    return Result;
  }

  static FunctionCallReturnValue maybe() {
    FunctionCallReturnValue Result;
    Result.Value = Maybe;
    return Result;
  }

  static FunctionCallReturnValue fromName(llvm::StringRef Name) {
    FunctionCallReturnValue Result;

    if (Name == "NoOrDead")
      Result.Value = NoOrDead;
    else if (Name == "Maybe")
      Result.Value = Maybe;
    else if (Name == "Yes")
      Result.Value = Yes;
    else if (Name == "Dead")
      Result.Value = Dead;
    else if (Name == "Contradiction")
      Result.Value = Contradiction;
    else if (Name == "No")
      Result.Value = No;
    else if (Name == "YesOrDead")
      Result.Value = YesOrDead;
    else
      revng_abort();

    return Result;
  }

public:
  bool isContradiction() const { return Value == Contradiction; }

  void notAvailable() {
    // All the situations which embed the possibility of being an argument have
    // to go to a case that includes the possibility of *not* being an argument
    // due to the register being a callee-saved register
    switch (Value) {
    case NoOrDead:
    case Maybe:
      // These are fine
      break;

    case Yes:
      // Weaken Yes statement
      Value = Maybe;
      break;

    case No:
    case Contradiction:
    case YesOrDead:
    case Dead:
      revng_abort();
    }
  }

  void combine(const FunctionReturnValue &Other);

  bool isCompatibleWith(const FunctionReturnValue &Other) const {
    switch (Value) {
    case YesOrDead:
      switch (Other.Value) {
      case FunctionReturnValue::YesOrDead:
      case FunctionReturnValue::Maybe:
      case FunctionReturnValue::NoOrDead:
        return true;
      case FunctionReturnValue::No:
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    case NoOrDead:
      switch (Other.Value) {
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::No:
      case FunctionReturnValue::Maybe:
      case FunctionReturnValue::YesOrDead:
        return true;
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    case Maybe:
      switch (Other.Value) {
      case FunctionReturnValue::Maybe:
      case FunctionReturnValue::YesOrDead:
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::No:
        return true;
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    case Yes:
      switch (Other.Value) {
      case FunctionReturnValue::YesOrDead:
      case FunctionReturnValue::Maybe:
        return true;
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::No:
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    case Dead:
      switch (Other.Value) {
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::YesOrDead:
      case FunctionReturnValue::Maybe:
        return true;
      case FunctionReturnValue::No:
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    case Contradiction:
      return false;
    case No:
      switch (Other.Value) {
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::No:
      case FunctionReturnValue::Maybe:
        return true;
      case FunctionReturnValue::YesOrDead:
      case FunctionReturnValue::Contradiction:
        return false;
      }
      break;
    }

    revng_abort();
  }

  const char *valueName() const {
    switch (Value) {
    case NoOrDead:
      return "NoOrDead";
    case Maybe:
      return "Maybe";
    case Yes:
      return "Yes";
    case Dead:
      return "Dead";
    case Contradiction:
      return "Contradiction";
    case YesOrDead:
      return "YesOrDead";
    case No:
      return "No";
    }

    revng_abort();
  }

  Values value() const { return Value; }

  void dump() const { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << valueName();
  }
};

template<typename K, typename V>
inline V getOrDefault(const std::map<K, V> &Map, K Key) {
  auto It = Map.find(Key);
  if (It == Map.end())
    return V();
  else
    return It->second;
}

/// \brief Class containg the final results about all the analyzed functions
class FunctionsSummary {
public:
  struct FunctionRegisterDescription {
    FunctionRegisterArgument Argument;
    FunctionReturnValue ReturnValue;
  };

  struct FunctionCallRegisterDescription {
    FunctionCallRegisterArgument Argument;
    FunctionCallReturnValue ReturnValue;

    bool isCompatibleWith(const FunctionRegisterDescription &FRD) const {
      return (Argument.isCompatibleWith(FRD.Argument)
              and ReturnValue.isCompatibleWith(FRD.ReturnValue));
    }
  };

  struct FunctionDescription;

  // TODO: this is finalized stuff, should we use vectors/SmalMaps instead of
  //       maps?
  struct CallSiteDescription {
    CallSiteDescription(llvm::Instruction *Call, llvm::Value *Callee) :
      Call(Call), Callee(Callee) {}

    llvm::Instruction *Call;
    llvm::Value *Callee;

    using GlobalVariable = llvm::GlobalVariable;

    template<typename K, typename V>
    using map = std::map<K, V>;

    map<GlobalVariable *, FunctionCallRegisterDescription> RegisterSlots;

    llvm::GlobalVariable *
    isCompatibleWith(const FunctionDescription &Function) const;
  };

  struct FunctionDescription {
    FunctionDescription() : Function(nullptr), Type(FunctionType::Invalid) {}

    llvm::Value *Function;
    FunctionType::Values Type;
    std::map<llvm::BasicBlock *, BranchType::Values> BasicBlocks;
    // TODO: this should be a vector
    std::map<llvm::GlobalVariable *, FunctionRegisterDescription> RegisterSlots;
    std::deque<CallSiteDescription> CallSites;
    std::set<llvm::GlobalVariable *> ClobberedRegisters;
  };

public:
  /// \brief Map from function entry points to its description
  std::map<llvm::BasicBlock *, FunctionDescription> Functions;

public:
  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  /// \brief Dump in JSON format
  ///
  /// [
  ///   {
  ///     "entry_point": "bb.main",
  ///     "entry_point_address": "0x1234",
  ///     "type": "type",
  ///     "reasons": ["Callee", "Direct", ...],
  ///     "basic_blocks": [
  ///       {
  ///         "name": "...",
  ///         "type": "...",
  ///         "start": "0x4444",
  ///         "end": "0x4488"
  ///       },
  ///       ...
  ///     ],
  ///     "slots": [
  ///       {
  ///         "slot": "CPU+rax",
  ///         "argument": "Dead",
  ///         "return_value": "Dead"
  ///       },
  ///       ...
  ///     ],
  ///     "range": [{"start": "0x5555", "end": "0x6666"}, ...],
  ///     "function_calls": [
  ///       {
  ///         "caller": "bb.callee:12",
  ///         "caller_address": "0x4567",
  ///         "slots": [
  ///           {
  ///             "slot": "CPU+rax",
  ///             "argument": "Dead",
  ///             "return_value": "Dead"
  ///           },
  ///           ...
  ///         ],
  ///       }
  ///     ],
  ///   }
  /// ]
  ///
  /// \note The functions are sorted according to entry_point. reasons are
  ///       are sorted too.
  template<typename O>
  void dump(const llvm::Module *M, O &Output) const {
    dumpInternal(M, StreamWrapper<O>(Output));
  }

private:
  void dumpInternal(const llvm::Module *M, StreamWrapperBase &&Stream) const;
};

} // namespace StackAnalysis

#endif // FUNCTIONSSUMMARY_H
