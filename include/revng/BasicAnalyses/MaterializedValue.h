#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

/// Class representing either a constant value or an offset from a symbol
class MaterializedValue {
public:
  class MemoryRange {
  public:
    MetaAddress Address;
    unsigned Size = 0;
  };

private:
  bool IsValid = false;
  std::optional<std::string> SymbolName;
  llvm::APInt Value;
  llvm::SmallVector<MemoryRange, 1> AffectedBy;

private:
  MaterializedValue(const llvm::APInt &Value) : IsValid(true), Value(Value) {}
  MaterializedValue(llvm::StringRef SymbolName, const llvm::APInt &Offset) :
    IsValid(true), SymbolName(SymbolName.str()), Value(Offset) {}

public:
  MaterializedValue() = default;
  MaterializedValue(const MaterializedValue &) = default;
  MaterializedValue(MaterializedValue &&) = default;
  MaterializedValue &operator=(const MaterializedValue &) = default;
  MaterializedValue &operator=(MaterializedValue &&) = default;

public:
  static MaterializedValue invalid() { return MaterializedValue(); }

  static MaterializedValue fromConstant(const llvm::APInt &API) {
    return MaterializedValue(API);
  }

  static MaterializedValue fromConstant(llvm::ConstantInt *CI);

  static MaterializedValue fromSymbol(llvm::StringRef SymbolName,
                                      const llvm::APInt &Offset) {
    return MaterializedValue(SymbolName, Offset);
  }

  static MaterializedValue apply(llvm::Instruction *Operation,
                                 llvm::ArrayRef<MaterializedValue> Operands);

public:
  template<typename T>
  MaterializedValue load(T &MemoryOracle, unsigned Size) const {
    // Cannot load symbols or invalid values
    if (hasSymbol() or not IsValid)
      return MaterializedValue::invalid();

    return MemoryOracle.load(Value.getLimitedValue(), Size);
  }

  MaterializedValue byteSwap() const {
    MaterializedValue Result = *this;
    Result.Value = Value.byteSwap();
    return Result;
  }

public:
  bool operator==(const MaterializedValue &Other) const {
    auto MaxBitWidth = std::max(Value.getBitWidth(), Other.Value.getBitWidth());
    auto ZExtValue = Value.zext(MaxBitWidth);
    auto ZExtOther = Other.Value.zext(MaxBitWidth);
    auto This = std::tie(IsValid, SymbolName, ZExtValue);
    auto That = std::tie(Other.IsValid, Other.SymbolName, ZExtOther);
    return This == That;
  }

  bool operator<(const MaterializedValue &Other) const {
    auto MaxBitWidth = std::max(Value.getBitWidth(), Other.Value.getBitWidth());
    if (Value.zext(MaxBitWidth).ult(Other.Value.zext(MaxBitWidth)))
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

  llvm::Constant *toConstant(llvm::LLVMContext &Context) const;

  llvm::Constant *toConstant(llvm::Type *DestinationType) const;

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
