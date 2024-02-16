#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <map>

#include "revng/Support/Debug.h"

extern Logger<> MarkLog;

namespace MarkAssignments {

namespace Reasons {

/// Bitmask that represents reasons why an Instruction needs an assignment.
enum Values {
  None = 0,

  // ==== Semantic Reasons ====
  // If an Instruction needs an assignment for one of this reasons, it means
  // that not emitting an assignment for it would affect semantics because of
  // side effects

  // The Instruction has side effects
  HasSideEffects = 1 << 0,
  // The Instruction does not have side effects per-se, but it might see the
  // side effects of other Instructions that do.
  // As an example, a LoadInst does not have side-effects when emitted in C,
  // but it could read a memory location where a previous StoreInst has written.
  HasInterferingSideEffects = 1 << 1,

  // ==== Cosmetic Reasons ====
  // If an Instruction needs an assignments for one of this reasons, it can
  // safely be ignored, without affecting semantics.

  // Failsafe flag to forcibly assign an Instruction with no uses to a variable.
  // This is useful for debug purposes, because it makes dead Instructions show
  // up in decompiled code.
  AlwaysAssign = 1 << 2,

  // TODO: this flag is used to to forcibly assign an Instruction if it's known
  // to occur often in dataflows that cause exponential blowup of the generated
  // string when emitting C code.
  // At the moment as a workaround we force them to be assigned to a local
  // variable, but the proper fix would be a visit on the dataflow graph to
  // actually solve the problem in a general and robust way.
  PreventExponentialStrings = 1 << 3,
};

} // end namespace Reasons

} // end namespace MarkAssignments

inline MarkAssignments::Reasons::Values
operator|(MarkAssignments::Reasons::Values A,
          MarkAssignments::Reasons::Values B) {
  using EnumT = decltype(A);
  using IntT = std::underlying_type_t<EnumT>;
  return static_cast<EnumT>(static_cast<IntT>(A) | static_cast<IntT>(B));
}

inline MarkAssignments::Reasons::Values
operator&(MarkAssignments::Reasons::Values A,
          MarkAssignments::Reasons::Values B) {
  using EnumT = decltype(A);
  using IntT = std::underlying_type_t<EnumT>;
  return static_cast<EnumT>(static_cast<IntT>(A) & static_cast<IntT>(B));
}

inline MarkAssignments::Reasons::Values
operator~(MarkAssignments::Reasons::Values A) {
  using EnumT = decltype(A);
  using IntT = std::underlying_type_t<EnumT>;
  return static_cast<EnumT>(~static_cast<IntT>(A));
}

namespace MarkAssignments {

/// Wrapper class for MarkAssignments::Reasons::Values
class Flags {
private:
  MarkAssignments::Reasons::Values TheFlags = Reasons::None;

public:
  Flags(MarkAssignments::Reasons::Values X) : TheFlags(X){};

  Flags() = default;
  ~Flags() = default;

  Flags(const Flags &) = default;
  Flags &operator=(const Flags &) = default;

  Flags(Flags &&) = default;
  Flags &operator=(Flags &&) = default;

  void set(Reasons::Values Flag) {
    TheFlags = static_cast<decltype(TheFlags)>(TheFlags | Flag);
  }

  bool isSet(Reasons::Values Flag) const { return TheFlags & Flag; }

  Reasons::Values value() const { return TheFlags; }

  explicit operator bool() const { return TheFlags != Reasons::None; }
};

} // end namespace MarkAssignments

namespace llvm {

class Instruction;
class Function;

} // namespace llvm

namespace MarkAssignments {

using AssignmentMap = std::map<llvm::Instruction *, Flags>;

/// Selects in F the Instructions that need an assignment when decompiling to C.
/// Returns a map with the selected Instruction, and the reasons of selection.
AssignmentMap selectAssignments(llvm::Function &);

} // end namespace MarkAssignments
