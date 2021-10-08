#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {

class Instruction;

} // namespace llvm

/// \brief bitmask that represents reasons why an Instruction muse be serialized
enum SerializationReason {
  None = 0,
  AlwaysSerialize = 1 << 0,
  HasSideEffects = 1 << 1,
  HasInterferingSideEffects = 1 << 2,
  HasManyUses = 1 << 3,
  NeedsLocalVarToComputeExpr = 1 << 4,
  NeedsManyStatements = 1 << 5,
};

inline SerializationReason
operator|(SerializationReason a, SerializationReason b) {
  using underlying_t = std::underlying_type_t<SerializationReason>;
  return static_cast<SerializationReason>(static_cast<underlying_t>(a)
                                          | static_cast<underlying_t>(b));
}

inline SerializationReason
operator&(SerializationReason a, SerializationReason b) {
  using underlying_t = std::underlying_type_t<SerializationReason>;
  return static_cast<SerializationReason>(static_cast<underlying_t>(a)
                                          & static_cast<underlying_t>(b));
}

inline SerializationReason operator~(SerializationReason a) {
  using underlying_t = std::underlying_type_t<SerializationReason>;
  return static_cast<SerializationReason>(~static_cast<underlying_t>(a));
}

/// \brief Wrapper class for SerializationReason
class SerializationFlags {
public:
  SerializationFlags() = default;
  ~SerializationFlags() = default;
  SerializationFlags(SerializationReason X) : Flags(X){};
  SerializationFlags(const SerializationFlags &) = default;
  SerializationFlags(SerializationFlags &&) = default;
  SerializationFlags &operator=(const SerializationFlags &) = default;
  SerializationFlags &operator=(SerializationFlags &&) = default;

  void set(SerializationReason Flag) {
    Flags = static_cast<decltype(Flags)>(Flags | Flag);
  }

  bool isSet(SerializationReason Flag) const { return Flags & Flag; }

  SerializationReason value() const { return Flags; }

  /// \brief Returns true if the Instruction associated with \F must be
  /// serialized in C.
  static bool mustBeSerialized(const SerializationFlags &F) {
    // If any of the bits is set, it must be serialized.
    return F.Flags != None;
  }

  /// \brief Returns true if the Instruction associated with \F needs a VarDecl
  /// in C.
  static bool needsVarDecl(const SerializationFlags &F) {

    // If it does not need serialization, it doesn't need a VarDecl either
    if (not mustBeSerialized(F))
      return false;

    // If the AlwaysSerialize bit is set, the instruction has no uses, so it
    // must be serialized but it doesn't need a VarDecl
    if (F.isSet(AlwaysSerialize))
      return false;

    // In all the other cases the instruction must be serialized and it has at
    // least one use, so we need a VarDecl to represent its value.
    return true;
  }

  /// \brief Returns true if the Instruction associated with \F is affected by
  /// side effects.
  static bool hasSideEffects(const SerializationFlags &F) {
    // HasSideEffects, and HasInterferingSideEffects imply side effects.
    return F.Flags & (HasSideEffects | HasInterferingSideEffects);
  }

  /// \brief Returns true if the Instruction associated with \F needs many
  /// statements in C.
  //
  // A notable example is InsertValue.
  static bool needsAdditionalStmts(const SerializationFlags &F) {
    return F.Flags & NeedsManyStatements;
  }

  explicit operator bool() const { return Flags != None; }

private:
  SerializationReason Flags = None;
};

using SerializationMap = std::map<const llvm::Instruction *,
                                  SerializationFlags>;
