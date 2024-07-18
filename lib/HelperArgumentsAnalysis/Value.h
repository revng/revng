#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Debug.h"

namespace aua {

using llvm::SmallVector;

class Function;

class Value {
  friend class Context;

public:
  enum class Kind {
    Invalid,
    AnyOf,
    Argument,
    Constant,
    BinaryOperator,
    FunctionOf
  };

protected:
  Kind TheKind = Kind::Invalid;
  SmallVector<const Value *, 2> Operands;

protected:
  Value(Kind TheKind, SmallVector<const Value *, 2> &&Values) :
    TheKind(TheKind), Operands(std::move(Values)) {}

public:
  ~Value() = default;

private:
  void deleteValue();

protected:
  void deduplicate() {
    llvm::sort(Operands);
    Operands.erase(std::unique(Operands.begin(), Operands.end()),
                   Operands.end());
  }

public:
  std::strong_ordering operator<=>(const Value &Other) const;
  bool operator==(const Value &Other) const = default;

public:
  template<typename T>
  SmallVector<const T *, 2> collect() const {
    SmallVector<const T *, 2> Result;

    SmallVector<const Value *, 2> Queue{ this };
    while (not Queue.empty()) {
      const Value *Current = Queue.pop_back_val();
      if (const T *Upcasted = llvm::dyn_cast<T>(Current))
        Result.push_back(Upcasted);

      for (const Value *Value : Current->Operands)
        Queue.push_back(Value);
    }

    return Result;
  }

  llvm::DenseSet<unsigned> collectArguments() const;

public:
  std::string toString() const;

public:
  auto kind() const { return TheKind; }
  const auto &operands() const { return Operands; }

public:
  void setOperands(SmallVector<const Value *, 2> &&NewOperands) {
    Operands = std::move(NewOperands);
  }

public:
  auto upcast(auto &&Handler);
  auto upcast(auto &&Handler) const;

public:
  std::size_t hash() const;

protected:
  std::strong_ordering compare(const Value &Other) const;

protected:
  std::string nodeName() const {
    return std::to_string(reinterpret_cast<uintptr_t>(this));
  }

  std::string node(const std::string &Label) const {
    return "  " + nodeName() + "[label=\"" + Label + "\"];\n";
  }

  std::string edgeTo(const Value &Other) const {
    return "  " + nodeName() + " -> " + Other.nodeName() + "\n";
  }

public:
  std::string toGraphPart() const;

  std::string toGraph() const;
};

class AnyOfValue : public Value {
  friend class Context;

private:
  AnyOfValue(SmallVector<const Value *, 2> &&Values) :
    Value(Kind::AnyOf, std::move(Values)) {
    deduplicate();
  }

public:
  static bool classof(const Value *B) { return B->kind() == Kind::AnyOf; }

public:
  std::strong_ordering operator<=>(const AnyOfValue &Other) const;

public:
  auto operands() const { return Operands; }

public:
  std::string toString() const;
  std::string toGraphPart() const;
};

class ArgumentValue : public Value {
  friend class Context;

private:
  unsigned Index = 0;

private:
  ArgumentValue(unsigned Index) : Value(Kind::Argument, {}), Index(Index) {}

public:
  static bool classof(const Value *B) { return B->kind() == Kind::Argument; }

public:
  std::strong_ordering operator<=>(const ArgumentValue &Other) const;

public:
  auto index() const { return Index; }
  std::size_t hash() const;

public:
  std::string toString() const { return "Argument" + std::to_string(Index); }
  std::string toGraphPart() const { return node(toString()); }
};

class ConstantValue : public Value {
  friend class Context;

private:
  uint64_t TheValue = 0;

public:
  ConstantValue(uint64_t TheValue) :
    Value(Kind::Constant, {}), TheValue(TheValue) {}

public:
  static bool classof(const Value *B) { return B->kind() == Kind::Constant; }

public:
  std::strong_ordering operator<=>(const ConstantValue &Other) const;
  std::size_t hash() const;

public:
  auto value() const { return TheValue; }

public:
  std::string toString() const { return std::to_string(TheValue); }
  std::string toGraphPart() const { return node(toString()); }
};

class BinaryOperatorValue : public Value {
  friend class Context;

public:
  enum Operator {
    Invalid,
    Add,
    Subtract,
    Multiply
  };

private:
  Operator Type = Invalid;

private:
  BinaryOperatorValue(Operator Type,
                      const Value &FirstValue,
                      const Value &SecondValue) :
    Value(Kind::BinaryOperator, { &FirstValue, &SecondValue }), Type(Type) {}

public:
  static bool classof(const Value *B) {
    return B->kind() == Kind::BinaryOperator;
  }

public:
  std::strong_ordering operator<=>(const BinaryOperatorValue &Other) const;

public:
  auto type() const { return Type; }
  const Value &firstOperand() const { return *Operands[0]; }
  const Value &secondOperand() const { return *Operands[1]; }

public:
  std::size_t hash() const;

public:
  std::string toString() const;

  std::string toGraphPart() const;

private:
  const char *symbol() const;
};

class FunctionOfValue : public Value {
  friend class Context;

private:
  FunctionOfValue(SmallVector<const Value *, 2> &&Values) :
    Value(Kind::FunctionOf, std::move(Values)) {}

public:
  static bool classof(const Value *B) { return B->kind() == Kind::FunctionOf; }

public:
  std::strong_ordering operator<=>(const FunctionOfValue &Other) const;

  bool isUnknown() const { return Operands.size() == 0; }

public:
  std::string toString() const;
  std::string toGraphPart() const;
};

inline auto Value::upcast(auto &&Handler) {
  switch (TheKind) {
    revng_abort();
  case Kind::AnyOf:
    return Handler(*llvm::cast<AnyOfValue>(this));
  case Kind::Argument:
    return Handler(*llvm::cast<ArgumentValue>(this));
  case Kind::Constant:
    return Handler(*llvm::cast<ConstantValue>(this));
  case Kind::BinaryOperator:
    return Handler(*llvm::cast<BinaryOperatorValue>(this));
  case Kind::FunctionOf:
    return Handler(*llvm::cast<FunctionOfValue>(this));
  case Kind::Invalid:
  default:
    revng_abort();
  }
}

auto Value::upcast(auto &&Handler) const {
  switch (TheKind) {
  case Kind::AnyOf:
    return Handler(*llvm::cast<const AnyOfValue>(this));
  case Kind::Argument:
    return Handler(*llvm::cast<const ArgumentValue>(this));
  case Kind::Constant:
    return Handler(*llvm::cast<const ConstantValue>(this));
  case Kind::BinaryOperator:
    return Handler(*llvm::cast<const BinaryOperatorValue>(this));
  case Kind::FunctionOf:
    return Handler(*llvm::cast<const FunctionOfValue>(this));
  case Kind::Invalid:
  default:
    revng_abort();
  }
}

} // namespace aua
