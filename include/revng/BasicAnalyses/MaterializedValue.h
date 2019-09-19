#ifndef REVNG_MATERIALIZEDVALUE_H
#define REVNG_MATERIALIZEDVALUE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

// Local libraries includes
#include "revng/Support/Debug.h"

const uint64_t MaxMaterializedValues = (1 << 16);

/// \brief Class representing either a constant value or an offset from a symbol
class MaterializedValue {
private:
  bool IsValid;
  llvm::Optional<llvm::StringRef> SymbolName;
  uint64_t Value;

public:
  MaterializedValue() : IsValid(false), Value(0) {}
  MaterializedValue(uint64_t Value) : IsValid(true), Value(Value) {}
  MaterializedValue(llvm::StringRef Name, uint64_t Offset) :
    IsValid(true), SymbolName(Name), Value(Offset) {}

public:
  static MaterializedValue invalid() { return MaterializedValue(); }

public:
  bool operator==(const MaterializedValue &Other) const {
    auto This = std::tie(IsValid, SymbolName, Value);
    auto That = std::tie(Other.IsValid, Other.SymbolName, Other.Value);
    return This == That;
  }

  bool operator<(const MaterializedValue &Other) const {
    auto This = std::tie(IsValid, SymbolName, Value);
    auto That = std::tie(Other.IsValid, Other.SymbolName, Other.Value);
    return This < That;
  }

  uint64_t value() const {
    revng_assert(isValid());
    return Value;
  }
  bool isValid() const { return IsValid; }
  bool hasSymbol() const { return SymbolName.hasValue(); }
  llvm::StringRef symbolName() const {
    revng_assert(isValid());
    revng_assert(hasSymbol());
    return *SymbolName;
  }

  void dump() const debug_function {
    if (not isValid()) {
      dbg << "(invalid)";
    } else {
      if (hasSymbol())
        dbg << symbolName().data() << "+";
      dbg << "0x" << std::hex << value();
    }
  }
};

using MaterializedValues = std::vector<MaterializedValue>;

#endif // REVNG_MATERIALIZEDVALUE_H
