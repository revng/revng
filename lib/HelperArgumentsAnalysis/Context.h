#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <unordered_set>

#include "revng/ADT/RecursiveCoroutine.h"

#include "Value.h"

template<typename T>
bool compareByDereferencing(const T *LHS, const T *RHS) {
  revng_assert(LHS != nullptr);
  revng_assert(RHS != nullptr);
  return *LHS < *RHS;
}

template<typename T>
bool equalByDereferencing(const T *LHS, const T *RHS) {
  revng_assert(LHS != nullptr);
  revng_assert(RHS != nullptr);
  return *LHS == *RHS;
}

template<typename T>
struct HashByDereferencing {
  std::size_t operator()(const T *Value) const noexcept {
    revng_assert(Value != nullptr);
    return std::hash<T>(*Value);
  }
};

namespace aua {

class Context {
private:
  std::set<Value *, decltype(compareByDereferencing<Value>) *> Values;

public:
  Context() : Values(compareByDereferencing<Value>) {}

  ~Context() {
    // We manually delete the value to avoid having a virtual destructor in
    // Value. A virtual method would introduce a virtual table, which is
    // redundant given we use LLVM RTTI.
    for (Value *Entry : Values)
      Entry->deleteValue();
  }

private:
  template<typename T>
  const Value &get(T &&Object) {

    auto IsArgument = [](const Value *Value) {
      return llvm::isa<ArgumentValue>(Value);
    };
    auto IsFunctionOf = [](const Value *Value) {
      return llvm::isa<FunctionOfValue>(Value);
    };
    auto IsAnyOf = [](const Value *Value) {
      return llvm::isa<AnyOfValue>(Value);
    };

    switch (Object.kind()) {
    case Value::Kind::Invalid:
      revng_abort();
    case Value::Kind::AnyOf: {
      Object.deduplicate();

      if (llvm::any_of(Object.Operands, IsFunctionOf)) {
        SmallVector<const Value *, 2> Operands;
        llvm::copy(Object.template collect<ArgumentValue>(),
                   std::back_inserter(Operands));
        return get(FunctionOfValue(std::move(Operands)));
      }

      // Optimize AnyOf(AnyOf(...), ...)
      if (llvm::any_of(Object.Operands, IsAnyOf)) {
        SmallVector<const Value *, 2> MergedValues;

        for (const Value *Value : Object.Operands) {
          if (llvm::isa<AnyOfValue>(Value)) {
            llvm::copy(Value->Operands, std::back_inserter(MergedValues));
          } else {
            MergedValues.push_back(Value);
          }
        }

        return get(AnyOfValue(std::move(MergedValues)));
      }

    } break;
    case Value::Kind::Argument:
      break;
    case Value::Kind::Constant:
      break;
    case Value::Kind::BinaryOperator: {
      auto &BinaryOperator = llvm::cast<BinaryOperatorValue>(Object);
      auto *ConstantLHS = llvm::dyn_cast<ConstantValue>(Object.Operands[0]);
      auto *ConstantRHS = llvm::dyn_cast<ConstantValue>(Object.Operands[1]);

      switch (BinaryOperator.type()) {
      case BinaryOperatorValue::Invalid:
        revng_abort();

      case BinaryOperatorValue::Add:
      case BinaryOperatorValue::Subtract:
        if (ConstantLHS != nullptr and ConstantLHS->value() == 0) {
          return *BinaryOperator.Operands[1];
        } else if (ConstantRHS != nullptr and ConstantRHS->value() == 0) {
          return *BinaryOperator.Operands[0];
        } else if (ConstantLHS != nullptr and ConstantRHS != nullptr) {
          // TODO: should we consider zero- vs sign-extension?
          switch (BinaryOperator.type()) {
          case BinaryOperatorValue::Add:
            return get(ConstantValue(ConstantLHS->value()
                                     + ConstantRHS->value()));
          case BinaryOperatorValue::Subtract:
            return get(ConstantValue(ConstantLHS->value()
                                     - ConstantRHS->value()));
          default:
            revng_abort();
          }
        }
        break;
      case BinaryOperatorValue::Multiply:
        if (ConstantLHS != nullptr and ConstantLHS->value() == 0) {
          return *Object.Operands[0];
        } else if (ConstantLHS != nullptr and ConstantLHS->value() == 1) {
          return *Object.Operands[1];
        } else if (ConstantRHS != nullptr and ConstantRHS->value() == 0) {
          return *Object.Operands[1];
        } else if (ConstantRHS != nullptr and ConstantRHS->value() == 1) {
          return *Object.Operands[0];
        } else if (ConstantLHS != nullptr and ConstantRHS != nullptr) {
          return get(ConstantValue(ConstantLHS->value()
                                   * ConstantRHS->value()));
        }
        break;
      }

    } break;
    case Value::Kind::FunctionOf:
      Object.deduplicate();

      // Optimize FunctionOf(FunctionOf(...), ...)
      if (llvm::any_of(Object.Operands, IsFunctionOf)) {
        SmallVector<const Value *, 2> MergedValues;

        for (const Value *Value : Object.Operands) {
          if (llvm::isa<FunctionOfValue>(Value)) {
            llvm::copy(Value->Operands, std::back_inserter(MergedValues));
          } else {
            MergedValues.push_back(Value);
          }
        }

        return get(FunctionOfValue(std::move(MergedValues)));
      }

      if (not llvm::all_of(Object.Operands, IsArgument)) {
        auto Result = Object;

        llvm::DenseSet<const ArgumentValue *> UsedArguments;
        for (const Value *Operand : Object.Operands) {
          if (auto *Argument = llvm::dyn_cast<ArgumentValue>(Operand)) {
            UsedArguments.insert(Argument);
          } else {
            for (const ArgumentValue *Argument :
                 Operand->collect<ArgumentValue>()) {
              UsedArguments.insert(Argument);
            }
          }
        }

        Result.Operands.clear();
        for (const ArgumentValue *Argument : UsedArguments)
          Result.Operands.push_back(Argument);

        return get(Result);
      }
      break;
    }

    auto It = Values.find(&Object);

    // If it's the first time we see this object, copy it on the heap
    if (It == Values.end()) {
      It = Values.insert(It, Object.upcast([](auto &Upcasted) -> Value * {
        using P = std::decay_t<decltype(Upcasted)>;
        return new P(Upcasted);
      }));
    }

    return **It;
  }

public:
  const Value &getAnyOf(SmallVector<const Value *, 2> &&Values) {
    return get(AnyOfValue(std::move(Values)));
  }

  const Value &getUnknown() { return getFunctionOf({}); }

  const Value &getFunctionOf(SmallVector<const Value *, 2> &&Values) {
    return get(FunctionOfValue(std::move(Values)));
  }

  const Value &getArgument(uint64_t Index) { return get(ArgumentValue(Index)); }

  const Value &getConstant(uint64_t Value) { return get(ConstantValue(Value)); }

  const Value &getAdd(const Value &FirstValue, const Value &SecondValue) {
    return get(BinaryOperatorValue(BinaryOperatorValue::Add,
                                   FirstValue,
                                   SecondValue));
  }

  const Value &getSubtract(const Value &FirstValue, const Value &SecondValue) {
    return get(BinaryOperatorValue(BinaryOperatorValue::Subtract,
                                   FirstValue,
                                   SecondValue));
  }

  const Value &getMultiply(const Value &FirstValue, const Value &SecondValue) {
    return get(BinaryOperatorValue(BinaryOperatorValue::Multiply,
                                   FirstValue,
                                   SecondValue));
  }

public:
  RecursiveCoroutine<const Value *>
  replaceArguments(const Value &Original,
                   const llvm::DenseMap<uint64_t, const Value *> &NewArguments);
};

} // namespace aua
