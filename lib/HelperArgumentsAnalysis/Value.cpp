/// \file Value.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "Value.h"

inline std::size_t hashCombine(std::size_t LHS, std::size_t RHS) {
  return LHS ^ (RHS << 1);
}

// TODO: this won't be necessary once we update libc++
template<class I1>
constexpr auto lexicographicalCompareThreeWay(I1 LeftBegin,
                                              I1 LeftEnd,
                                              I1 RightBegin,
                                              I1 RigthEnd)
  -> decltype(std::compare_three_way()(*LeftBegin, *RightBegin)) {
  using ret_t = decltype(std::compare_three_way()(*LeftBegin, *RightBegin));
  static_assert(std::disjunction_v<std::is_same<ret_t, std::strong_ordering>,
                                   std::is_same<ret_t, std::weak_ordering>,
                                   std::is_same<ret_t, std::partial_ordering>>,
                "The return type must be a comparison category type.");

  bool Exhaust1 = (LeftBegin == LeftEnd);
  bool Exhaust2 = (RightBegin == RigthEnd);
  for (; !Exhaust1 && !Exhaust2; Exhaust1 = (++LeftBegin == LeftEnd),
                                 Exhaust2 = (++RightBegin == RigthEnd))
    if (auto C = std::compare_three_way()(*LeftBegin, *RightBegin); C != 0)
      return C;

  return !Exhaust1 ? std::strong_ordering::greater :
         !Exhaust2 ? std::strong_ordering::less :
                     std::strong_ordering::equal;
}

namespace aua {

std::size_t Value::hash() const {
  std::size_t Result = std::hash<Value::Kind>()(TheKind);
  for (const Value *V : Operands)
    Result = hashCombine(Result, std::hash<const Value *>()(V));
  return Result;
}

std::strong_ordering Value::compare(const Value &Other) const {
  return lexicographicalCompareThreeWay(Operands.begin(),
                                        Operands.end(),
                                        Other.Operands.begin(),
                                        Other.Operands.end());
}

std::string Value::toGraph() const {
  std::string Result;
  Result += "digraph {\n";
  Result += "  node[shape=box];\n";
  Result += "  rankdir = BT;\n";
  Result += toGraphPart();
  Result += "}\n";
  return Result;
}

std::strong_ordering AnyOfValue::operator<=>(const AnyOfValue &Other) const {
  std::strong_ordering Result = compare(Other);
  if (Result == std::strong_ordering::equal) {
    return Result;
  } else {
    return Result;
  }
}

std::string AnyOfValue::toString() const {
  std::string Result = "AnyOf(";
  const char *Separator = "";
  for (const Value *Value : Operands) {
    Result += Separator;
    Result += Value->toString();
    Separator = ", ";
  }
  Result += ")";
  return Result;
}

std::string AnyOfValue::toGraphPart() const {
  std::string Result = node("AnyOf");

  for (const Value *Value : Operands) {
    Result += Value->toGraphPart();
    Result += edgeTo(*Value);
  }

  return Result;
}

std::strong_ordering
ArgumentValue::operator<=>(const ArgumentValue &Other) const {
  std::strong_ordering Result = compare(Other);
  if (Result == std::strong_ordering::equal) {
    return Index <=> Other.Index;
  } else {
    return Result;
  }
}

std::size_t ArgumentValue::hash() const {
  return hashCombine(Value::hash(), std::hash<unsigned>()(Index));
}

std::strong_ordering
ConstantValue::operator<=>(const ConstantValue &Other) const {
  std::strong_ordering Result = compare(Other);
  if (Result == std::strong_ordering::equal) {
    return TheValue <=> Other.TheValue;
  } else {
    return Result;
  }
}

std::size_t ConstantValue::hash() const {
  return hashCombine(Value::hash(), std::hash<uint64_t>()(TheValue));
}

std::strong_ordering
BinaryOperatorValue::operator<=>(const BinaryOperatorValue &Other) const {
  std::strong_ordering Result = compare(Other);
  if (Result == std::strong_ordering::equal) {
    return Type <=> Other.Type;
  } else {
    return Result;
  }
}

std::size_t BinaryOperatorValue::hash() const {
  return hashCombine(Value::hash(), std::hash<Operator>()(Type));
}

std::string BinaryOperatorValue::toString() const {
  return "(" + Operands[0]->toString() + " " + symbol() + " "
         + Operands[1]->toString() + ")";
}

std::string BinaryOperatorValue::toGraphPart() const {
  std::string Result;
  Result += node(symbol());
  Result += Operands[0]->toGraphPart();
  Result += edgeTo(*Operands[0]);
  Result += Operands[1]->toGraphPart();
  Result += edgeTo(*Operands[1]);
  return Result;
}

const char *BinaryOperatorValue::symbol() const {
  switch (Type) {
  case Add:
    return "+";
  case Subtract:
    return "-";
  case Multiply:
    return "*";
  default:
  case Invalid:
    revng_abort();
  }
}

std::strong_ordering
FunctionOfValue::operator<=>(const FunctionOfValue &Other) const {
  std::strong_ordering Result = compare(Other);
  if (Result == std::strong_ordering::equal) {
    return Result;
  } else {
    return Result;
  }
}

std::string FunctionOfValue::toString() const {
  std::string Result = "FunctionOf(";
  const char *Separator = "";
  for (const Value *Value : Operands) {
    Result += Separator;
    Result += Value->toString();
    Separator = ", ";
  }
  Result += ")";
  return Result;
}

std::string FunctionOfValue::toGraphPart() const {
  std::string Result = node("FunctionOf");

  for (const Value *Value : Operands) {
    Result += Value->toGraphPart();
    Result += edgeTo(*Value);
  }

  return Result;
}

llvm::DenseSet<unsigned> Value::collectArguments() const {
  llvm::DenseSet<unsigned> Result;
  for (auto &Argument : collect<ArgumentValue>())
    Result.insert(Argument->index());
  return Result;
}

void Value::deleteValue() {
  upcast([](auto &Upcasted) { delete &Upcasted; });
}

std::strong_ordering Value::operator<=>(const Value &Other) const {
  if (TheKind != Other.TheKind)
    return TheKind <=> Other.TheKind;

  return upcast([&Other](auto &Upcasted) {
    using Type = std::decay_t<decltype(Upcasted)>;
    return Upcasted <=> *(llvm::cast<const Type>(&Other));
  });
}

std::string Value::toString() const {
  return upcast([](auto &Upcasted) { return Upcasted.toString(); });
}

std::string Value::toGraphPart() const {
  return upcast([](const auto &Upcasted) { return Upcasted.toGraphPart(); });
}
} // namespace aua
