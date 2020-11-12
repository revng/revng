#ifndef REVNG_MATERIALIZEDVALUE_H
#define REVNG_MATERIALIZEDVALUE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Debug.h"

/// \brief Class representing either a constant value or an offset from a symbol
class MaterializedValue {
private:
  bool IsValid;
  llvm::Optional<llvm::StringRef> SymbolName;
  llvm::APInt Value;

public:
  MaterializedValue() : IsValid(false), Value() {}
  MaterializedValue(const llvm::APInt &Value) : IsValid(true), Value(Value) {}
  MaterializedValue(llvm::StringRef Name, const llvm::APInt &Offset) :
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
    if (Value.ult(Other.Value))
      return true;
    auto This = std::tie(IsValid, SymbolName);
    auto That = std::tie(Other.IsValid, Other.SymbolName);
    return This < That;
  }

  const llvm::APInt &value() const {
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

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    if (not isValid()) {
      Output << "(invalid)";
    } else {
      if (hasSymbol())
        Output << symbolName().data() << "+";
      llvm::SmallString<40> String;
      value().toStringUnsigned(String, 16);
      Output << "0x" << String.c_str();
    }
  }
};

using MaterializedValues = std::vector<MaterializedValue>;

#endif // REVNG_MATERIALIZEDVALUE_H
