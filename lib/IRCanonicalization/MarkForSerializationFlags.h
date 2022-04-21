#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

namespace llvm {

class Instruction;

} // namespace llvm

/// bitmask that represents reasons why an Instruction muse be serialized
enum SerializationReason {
  None = 0,
  AlwaysSerialize = 1 << 0,
  HasSideEffects = 1 << 1,
  HasInterferingSideEffects = 1 << 2,
  HasManyUses = 1 << 3,
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

/// Wrapper class for SerializationReason
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

  /// Returns true if the Instruction associated with \F is affected by
  /// side effects.
  static bool hasSideEffects(const SerializationFlags &F) {
    // HasSideEffects, and HasInterferingSideEffects imply side effects.
    return F.Flags & (HasSideEffects | HasInterferingSideEffects);
  }

  explicit operator bool() const { return Flags != None; }

private:
  SerializationReason Flags = None;
};

using SerializationMap = std::map<llvm::Instruction *, SerializationFlags>;
