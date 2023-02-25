#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Debug.h"

/// Class representing either a constant value or an offset from a symbol
class MaterializedValue {
private:
  bool IsValid;
  std::optional<std::string> SymbolName;
  llvm::APInt Value;

public:
  MaterializedValue() : IsValid(false), Value() {}
  MaterializedValue(const llvm::APInt &Value) : IsValid(true), Value(Value) {}
  MaterializedValue(std::string Name, const llvm::APInt &Offset) :
    IsValid(true), SymbolName(Name), Value(Offset) {}

public:
  static MaterializedValue invalid() { return MaterializedValue(); }

public:
  bool operator==(const MaterializedValue &Other) const {
    auto MaxBitWidth = std::max(Value.getBitWidth(), Other.Value.getBitWidth());
    auto ZExtValue = Value.zextOrSelf(MaxBitWidth);
    auto ZExtOther = Other.Value.zextOrSelf(MaxBitWidth);
    auto This = std::tie(IsValid, SymbolName, ZExtValue);
    auto That = std::tie(Other.IsValid, Other.SymbolName, ZExtOther);
    return This == That;
  }

  bool operator<(const MaterializedValue &Other) const {
    auto MaxBitWidth = std::max(Value.getBitWidth(), Other.Value.getBitWidth());
    if (Value.zextOrSelf(MaxBitWidth).ult(Other.Value.zextOrSelf(MaxBitWidth)))
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
  bool hasSymbol() const { return SymbolName.has_value(); }
  std::string symbolName() const {
    revng_assert(isValid());
    revng_assert(hasSymbol());
    revng_assert(not llvm::StringRef(*SymbolName).contains('\0'));
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
