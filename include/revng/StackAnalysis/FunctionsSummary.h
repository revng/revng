#ifndef FUNCTIONSSUMMARY_H
#define FUNCTIONSSUMMARY_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file is NOT automatically generated.

// Standard includes
#include <map>
#include <set>

// Local libraries includes
#include "revng/Support/Debug.h"

namespace llvm {
class BasicBlock;
class GlobalVariable;
class Instruction;
class Module;
} // namespace llvm

namespace StackAnalysis {

namespace FunctionType {

enum Values {
  Invalid, ///< An invalid entry
  Regular, ///< A normal function
  NoReturn, ///< A noreturn function
  IndirectTailCall, ///< A function with no returns but at least an indirect
                    ///  tail call
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
  case IndirectTailCall:
    return "IndirectTailCall";
  case Fake:
    return "Fake";
  }

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
  FunctionSummary,
  /// This is a function for which we couldn't find any return statement
  NoReturnFunction,
  /// This is a function for which there are no proper return statements, but
  /// not all exit points are killer/noreturns
  IndirectTailCallFunction
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
  case IndirectTailCall:
    return "IndirectTailCall";
  case FakeFunction:
    return "FakeFunction";
  case LongJmp:
    return "LongJmp";
  case Killer:
    return "Killer";
  case FunctionSummary:
    return "FunctionSummary";
  case NoReturnFunction:
    return "NoReturnFunction";
  case Unreachable:
    return "Unreachable";
  case IndirectTailCallFunction:
    return "IndirectTailCallFunction";
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
  else if (Name == "IndirectTailCall")
    return IndirectTailCall;
  else if (Name == "FakeFunction")
    return FakeFunction;
  else if (Name == "LongJmp")
    return LongJmp;
  else if (Name == "Killer")
    return Killer;
  else if (Name == "FunctionSummary")
    return FunctionSummary;
  else if (Name == "NoReturnFunction")
    return NoReturnFunction;
  else if (Name == "Unreachable")
    return Unreachable;
  else if (Name == "IndirectTailCallFunction")
    return IndirectTailCallFunction;
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
    case No:
    case NoOrDead:
    case Maybe:
    case Contradiction:
      // These are already good
      break;
    case Dead:
      // Weakend the statement
      Value = NoOrDead;
      break;
    case Yes:
      // Weakend the statement
      Value = Maybe;
      break;
    }
  }

  void combine(const RegisterArgument<not FunctionCall> &Other);

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
  enum Values { No, NoOrDead, Dead, Yes, Maybe, YesCandidate, Contradiction };

private:
  Values Value;

public:
  FunctionReturnValue() : Value(Maybe) {}

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
    case Yes:
      return "Yes";
    case Dead:
      return "Dead";
    case No:
      return "No";
    case YesCandidate:
      return "YesCandidate";
    case Contradiction:
      return "Contradiction";
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
  enum Values { No, NoOrDead, Maybe, Yes, Dead, Contradiction };

private:
  Values Value;

public:
  FunctionCallReturnValue() : Value(Maybe) {}

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

public:
  bool isContradiction() const { return Value == Contradiction; }

  void notAvailable() {
    // All the situations which embed the possibility of being an argument have
    // to go to a case that includes the possibility of *not* being an argument
    // due to the register being a callee-saved register
    switch (Value) {
    case No:
    case NoOrDead:
    case Maybe:
    case Contradiction:
      // These are fine
      break;

    case Yes:
      // Weaken Yes statement
      Value = Maybe;
      break;

    case Dead:
      // Weakend Dead statement
      Value = NoOrDead;
      break;
    }
  }

  void combine(const FunctionReturnValue &Other);

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
  };

  // TODO: this is finalized stuff, should we use vectors/SmalMaps instead of
  //       maps?
  struct CallSiteDescription {
    CallSiteDescription(llvm::Instruction *Call, llvm::BasicBlock *Callee) :
      Call(Call),
      Callee(Callee) {}

    llvm::Instruction *Call;
    llvm::BasicBlock *Callee;

    using GlobalVariable = llvm::GlobalVariable;

    template<typename K, typename V>
    using map = std::map<K, V>;

    map<GlobalVariable *, FunctionCallRegisterDescription> RegisterSlots;
  };

  struct FunctionDescription {
    FunctionDescription() : Type(FunctionType::Invalid) {}

    FunctionType::Values Type;
    std::map<llvm::BasicBlock *, BranchType::Values> BasicBlocks;
    std::map<llvm::GlobalVariable *, FunctionRegisterDescription> RegisterSlots;
    std::vector<CallSiteDescription> CallSites;
    std::set<llvm::GlobalVariable *> ClobberedRegisters;
  };

public:
  /// \brief Map from function entry points to its description
  std::map<llvm::BasicBlock *, FunctionDescription> Functions;

public:
  void dump(const llvm::Module *M) const { dump(M, dbg); }

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
