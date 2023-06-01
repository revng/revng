//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Generated/Early/TypeKind.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using llvm::AnalysisUsage;
using llvm::APInt;
using llvm::cast;
using llvm::Constant;
using llvm::ConstantExpr;
using llvm::ConstantInt;
using llvm::dyn_cast;
using llvm::FunctionPass;
using llvm::Instruction;
using llvm::IRBuilder;
using llvm::isa;
using llvm::LLVMContext;
using llvm::PHINode;
using llvm::RegisterPass;
using llvm::ReversePostOrderTraversal;
using llvm::SmallVector;
using llvm::Use;
using llvm::User;
using llvm::Value;
using model::CABIFunctionType;
using model::Qualifier;
using model::RawFunctionType;

static Logger<> ModelGEPLog{ "make-model-gep" };

// This struct represents an llvm::Value for which it has been determined that
// it has pointer semantic on the model, along with the model::QualifiedType of
// the pointee.
class ModelTypedIRAddress {

private:
  model::QualifiedType PointeeType = {};
  Value *Address = nullptr;

public:
  // The default constructor constructs an invalid ModelTypedIRAddress
  ModelTypedIRAddress() = default;

  ModelTypedIRAddress(const ModelTypedIRAddress &) = default;
  ModelTypedIRAddress &operator=(const ModelTypedIRAddress &) = default;

  ModelTypedIRAddress(ModelTypedIRAddress &) = default;
  ModelTypedIRAddress &operator=(ModelTypedIRAddress &) = default;

public:
  ModelTypedIRAddress(const model::QualifiedType &PT, Value *A) :
    PointeeType(PT), Address(A) {
    revng_assert(PointeeType.verify() and nullptr != Address);
  }

  static ModelTypedIRAddress invalid() { return ModelTypedIRAddress(); }

  bool isValid() const {
    return nullptr != Address and PointeeType.UnqualifiedType().isValid()
           and not PointeeType.UnqualifiedType().empty();
  }

  const auto &getPointeeType() const { return PointeeType; }
  const auto &getAddress() const { return Address; }

  // Enable structured bindings, but only the const version, so that the
  // internal state cannot be inadvertently changed (and possibly invalidated)
  // through bindings.
  template<int I>
  auto const &get() const {
    if constexpr (I == 0)
      return PointeeType;
    else if constexpr (I == 1)
      return Address;
    else
      static_assert(value_always_false_v<I>);
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelTypedIRAddress = ";
    if (not isValid()) {
      OS << "invalid";
      return;
    }

    OS << "{\nPointeeType:\n";
    serialize(OS, PointeeType);

    OS << "\nAddress: ";
    Address->print(OS);

    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

// Enable structured bindings for ModelTypedIRAddress
namespace std {

template<>
struct tuple_size<ModelTypedIRAddress> : std::integral_constant<size_t, 2> {};

template<int I>
struct tuple_element<I, ModelTypedIRAddress> {
  using GetReturnType = decltype(std::declval<ModelTypedIRAddress>().get<I>());
  using type = std::remove_reference_t<GetReturnType>;
};

} // end namespace std

static std::string toDecimal(const APInt &Number) {
  llvm::SmallString<16> Result;
  Number.toString(Result, 10, true);
  return Result.str().str();
}

static ConstantInt *getOneWithSameType(const Value *IntOrPtr) {
  llvm::Type *TheType = IntOrPtr->getType();
  revng_assert(TheType->isIntOrPtrTy());
  unsigned BitWidth = TheType->isIntegerTy() ? TheType->getIntegerBitWidth() :
                                               64;

  APInt TheOne = APInt(/*NumBits*/ BitWidth, /*Value*/ 1);
  return ConstantInt::get(IntOrPtr->getContext(), TheOne);
}

// This struct represents an addend of a summation on the LLVM IR in the form
// Coefficient * Index, where Coefficient is constant and Index can be whatever
// but not a constant.
class IRAddend {

private:
  ConstantInt *Coefficient;
  Value *Index;

public:
  // Delete default constructor to forbid creation of invalid IRAddend.
  IRAddend() = delete;

  // IRAddends can be created only with this constructor, that ensures they're
  // always valid.
  IRAddend(ConstantInt *C, Value *I) : Coefficient(C), Index(I) {
    revng_assert(nullptr != C and nullptr != I);
    // The index must always be non-constant
    revng_assert(not isa<Constant>(I));
    // The index must always be an integer or a pointer
    revng_assert(I->getType()->isIntOrPtrTy());
    // The Coefficient must be non negative
    revng_assert(C->getValue().isNonNegative());
  }
  IRAddend(Value *V) : IRAddend(getOneWithSameType(V), V) {}

  IRAddend(const IRAddend &) = default;
  IRAddend &operator=(const IRAddend &) = default;

  IRAddend(IRAddend &&) = default;
  IRAddend &operator=(IRAddend &&) = default;

  // Accessors that prevents mutations.
  const auto &coefficient() const { return Coefficient; }
  const auto &index() const { return Index; }

  // Enable structured bindings, but only the const version, so that the
  // internal state cannot be inadvertently changed (and possibly invalidated)
  // through structured bindings.
  template<int I>
  auto const &get() const {
    if constexpr (I == 0)
      return Coefficient;
    else if constexpr (I == 1)
      return Index;
    else
      static_assert(value_always_false_v<I>);
  }

  // Debug prints
  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "IRAddend {\nCofficient:\n";
    OS << toDecimal(Coefficient->getValue());
    OS << "\nIndex:\n";
    Index->print(OS);
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

// Enable structured bindings for IRAddend
namespace std {

template<>
struct tuple_size<IRAddend> : std::integral_constant<size_t, 2> {};

template<int I>
struct tuple_element<I, IRAddend> {
  using GetReturnType = decltype(std::declval<IRAddend>().get<I>());
  using type = std::remove_reference_t<GetReturnType>;
};

} // end namespace std

static bool coefficientUGT(const APInt &LHS, const APInt &RHS) {
  revng_assert(LHS.isNonNegative());
  revng_assert(RHS.isNonNegative());
  auto MaxBitWidth = std::max(LHS.getBitWidth(), RHS.getBitWidth());
  APInt LargeLHS = LHS.zext(MaxBitWidth);
  APInt LargeRHS = RHS.zext(MaxBitWidth);
  return LargeLHS.ugt(LargeRHS);
};

static bool hasCoefficientUGT(const IRAddend &LHS, const IRAddend &RHS) {
  return coefficientUGT(LHS.coefficient()->getValue(),
                        RHS.coefficient()->getValue());
}

static bool hasCoefficientULT(const IRAddend &LHS, const IRAddend &RHS) {
  return coefficientUGT(RHS.coefficient()->getValue(),
                        LHS.coefficient()->getValue());
};

class IRSummation {
private:
  static APInt getZeroAPInt() { return APInt(/*NumBits*/ 64, /*Value*/ 0); }

  APInt Constant = getZeroAPInt();
  SmallVector<IRAddend> Indices = {};

public:
  // Construct from constants
  explicit IRSummation(size_t C) : Constant(APInt(64, C)), Indices(){};
  explicit IRSummation(APInt C) : Constant(C), Indices(){};
  IRSummation() : IRSummation(getZeroAPInt()){};

  // Construct from vectors of indices
  explicit IRSummation(APInt C, const SmallVector<IRAddend> &I) :
    Constant(C), Indices(I) {}
  explicit IRSummation(APInt C, SmallVector<IRAddend> &&I) :
    Constant(C), Indices(std::move(I)) {}

  explicit IRSummation(const SmallVector<IRAddend> &I) :
    IRSummation(getZeroAPInt(), I) {}
  explicit IRSummation(SmallVector<IRAddend> &&I) :
    IRSummation(getZeroAPInt(), std::move(I)) {}

  // Construct from single addends
  explicit IRSummation(APInt C, const IRAddend &Addend) :
    IRSummation(C, SmallVector<IRAddend>{ Addend }) {}
  explicit IRSummation(APInt C, IRAddend &&Addend) :
    IRSummation(C, SmallVector<IRAddend>{ std::move(Addend) }) {}

  explicit IRSummation(const IRAddend &Addend) :
    IRSummation(getZeroAPInt(), Addend) {}
  explicit IRSummation(IRAddend &&Addend) :
    IRSummation(getZeroAPInt(), std::move(Addend)) {}

  // Construct from single non-constant Value
  explicit IRSummation(Value *V) : IRSummation(IRAddend(V)) {}

public:
  IRSummation(const IRSummation &) = default;
  IRSummation &operator=(const IRSummation &) = default;

  IRSummation(IRSummation &&) = default;
  IRSummation &operator=(IRSummation &&) = default;

  // Helpers

  bool isValid() const debug_function {
    // Coefficients should be non-negative and increasing.
    std::optional<APInt> Previous = std::nullopt;
    for (const auto &I : Indices) {
      APInt Current = I.coefficient()->getValue();
      if (Current.isNegative())
        return false;
      if (Previous.has_value()) {
        auto MaxBitWidth = std::max(Previous->getBitWidth(),
                                    Current.getBitWidth());
        if (Current.zext(MaxBitWidth).ugt(Previous->zext(MaxBitWidth)))
          return false;
      }
      Previous = Current;
    }
    return true;
  }

  bool isConstant() const {
    revng_assert(isValid());
    return Indices.empty();
  }

  bool isZero() const { return Indices.empty() and Constant.isZero(); }

  const auto &getConstant() const { return Constant; }
  const auto &getIndices() const { return Indices; }

  // Enable structured bindings, but only the const version, so that the
  // internal state cannot be inadvertently changed (and possibly invalidated)
  // through bindings.
  template<int I>
  auto const &get() const {
    if constexpr (I == 0)
      return Constant;
    else if constexpr (I == 1)
      return Indices;
    else
      static_assert(value_always_false_v<I>);
  }

  // Arithmetic operators
  IRSummation operator+(const IRSummation &RHS) const {
    IRSummation Result = *this;
    if (this->Constant.getBitWidth() == RHS.Constant.getBitWidth()) {
      Result.Constant += RHS.Constant;
    } else {
      auto MaxBitWidth = std::max(this->Constant.getBitWidth(),
                                  RHS.Constant.getBitWidth());
      Result.Constant = Result.Constant.zext(MaxBitWidth)
                        + RHS.Constant.zext(MaxBitWidth);
    }
    revng_assert(not Result.Constant.isNegative());
    Result.Indices.append(RHS.Indices);
    llvm::sort(Result.Indices, hasCoefficientUGT);
    revng_assert(Result.isValid());
    return Result;
  }

  IRSummation &operator+=(const IRSummation &RHS) {
    *this = *this + RHS;
    return *this;
  }

  IRSummation operator-(size_t RHS) const {
    revng_assert(this->Constant.uge(RHS),
                 "Subtracting offset from IRSummation would underflow");
    IRSummation Result = *this;
    Result.Constant -= RHS;
    return Result;
  }

  IRSummation &operator-=(size_t RHS) {
    *this = *this - RHS;
    return *this;
  }

  bool operator==(const IRSummation &) const = default;

  // Debug prints
  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "IRSummation = {" << toDecimal(Constant);

    for (const auto &I : Indices) {
      OS << " + ";
      I.dump(OS);
    }

    OS << "}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

// Enable structured bindings for IRSummation
namespace std {

template<>
struct tuple_size<IRSummation> : std::integral_constant<size_t, 2> {};

template<int I>
struct tuple_element<I, IRSummation> {
  using GetReturnType = decltype(std::declval<IRSummation>().get<I>());
  using type = std::remove_reference_t<GetReturnType>;
};

} // end namespace std

// This struct represents an "expression" of the form:
//     BaseAddress + sum( const_i * index_i)
// This "expression" is actually tied to the LLVM IR, because const_i is an
// llvm::ConstantInt, index_i is an llvm::Value.
// When BaseAddress holds a valid value, it's a pointer to the llvm::Value
// representing the base address of the pointer arithmetic along with its model
// type. Otherwise it represents zero, so there's only the Summation term and
// the arithmetic is plain integer.
struct IRArithmetic {

  // If this is nullopt this summation does not represent an address, but simply
  // a summation of offsets.
  ModelTypedIRAddress BaseAddress = {};

  // If this is empty it means a zero offset.
  IRSummation Summation = {};

  IRArithmetic() = default;

  IRArithmetic(const IRArithmetic &) = default;
  IRArithmetic &operator=(const IRArithmetic &) = default;

  IRArithmetic(IRArithmetic &&) = default;
  IRArithmetic &operator=(IRArithmetic &&) = default;

private:
  // Force use of factories for more advanced constructors

  // Construct an IRArithmetic with only the base address, and no actual
  // arithmetic.
  IRArithmetic(ModelTypedIRAddress &&BA) :
    BaseAddress(std::move(BA)), Summation() {}

  // Construct an IRArithmetic without base address, that represents only an
  // addend without
  IRArithmetic(IRAddend &&Addend) :
    BaseAddress(ModelTypedIRAddress::invalid()),
    Summation(IRSummation(std::move(Addend))) {}

  IRArithmetic(APInt Constant) :
    BaseAddress(ModelTypedIRAddress::invalid()),
    Summation(IRSummation(Constant)) {}

public:
  bool isAddress() const { return BaseAddress.isValid(); }

  //
  // Factories
  //

  static IRArithmetic address(model::QualifiedType Type, Value *Address) {
    return IRArithmetic(ModelTypedIRAddress(dropPointer(Type), Address));
  }

  static IRArithmetic index(ConstantInt *Coefficient, Value *Index) {
    // The summation has only one element, with a constant coefficient
    return IRArithmetic(IRAddend(Coefficient, Index));
  }

  static IRArithmetic constant(ConstantInt *C) {
    return IRArithmetic(C->getValue());
  }

  static IRArithmetic unknown(Value *Unknown) { return IRArithmetic(Unknown); }

  //
  // Debug prints
  //

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "IRArithmetic {\nBaseAddress: ";
    BaseAddress.dump(OS);
    OS << "\nSummation: {\n";
    Summation.dump(OS);
    OS << "\n}\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using ModelTypesMap = std::map<const Value *, const model::QualifiedType>;

static RecursiveCoroutine<std::optional<IRArithmetic>>
getIRArithmetic(Use &AddressUse, const ModelTypesMap &PointerTypes) {
  revng_log(ModelGEPLog,
            "getIRArithmetic for use of: " << dumpToString(AddressUse.get()));
  LoggerIndent Indent{ ModelGEPLog };

  Value *AddressArith = AddressUse.get();
  // If the used value and we know it has a pointer type, we already know
  // both the base address and the pointer type.
  if (auto TypeIt = PointerTypes.find(AddressArith);
      TypeIt != PointerTypes.end()) {

    auto &[AddressVal, Type] = *TypeIt;

    revng_assert(Type.isPointer());
    revng_log(ModelGEPLog, "Use is typed!");

    rc_return IRArithmetic::address(Type, AddressArith);

  } else if (isa<ConstantExpr>(AddressArith)
             or isa<Instruction>(AddressArith)) {

    revng_log(ModelGEPLog, "Traverse cast!");
    auto *ConstantValue = dyn_cast<ConstantInt>(skipCasts(AddressArith));
    if (ConstantValue)
      rc_return IRArithmetic::constant(ConstantValue);

    auto *AddrArithmeticInst = dyn_cast<Instruction>(AddressArith);
    auto *ConstExprAddrArith = dyn_cast<ConstantExpr>(AddressArith);
    if (ConstExprAddrArith)
      AddrArithmeticInst = ConstExprAddrArith->getAsInstruction();

    switch (AddrArithmeticInst->getOpcode()) {

    case Instruction::Add: {
      auto *Add = cast<llvm::BinaryOperator>(AddrArithmeticInst);

      Use &LHSUse = Add->getOperandUse(0);
      auto LHSOrNone = rc_recur getIRArithmetic(LHSUse, PointerTypes);

      Use &RHSUse = Add->getOperandUse(1);
      auto RHSOrNone = rc_recur getIRArithmetic(RHSUse, PointerTypes);

      if (not RHSOrNone.has_value() or not LHSOrNone.has_value())
        break;

      auto &LHS = LHSOrNone.value();
      auto &RHS = RHSOrNone.value();

      bool LHSIsAddress = LHS.isAddress();
      bool RHSIsAddress = RHS.isAddress();

      if (LHSIsAddress and RHSIsAddress) {
        // In principle we should not expect to have many base address.
        // If we do, at the moment we don't have a better policy than to bail
        // out, and in principle this is totally safe, even if we give up a
        // chance to emit good model geps for this case.
        // In any case, we might want to devise smarter policies to
        // discriminate between different base addresses. Anyway it's not
        // clear if we can ever do something better than this.
        rc_return std::nullopt;
      }

      // Here at least one among LHS and RHS is not an address.

      IRArithmetic Result = LHSIsAddress ? LHS : RHS;
      Result.Summation += LHSIsAddress ? RHS.Summation : LHS.Summation;
      rc_return Result;
    } break;

    case Instruction::ZExt: {
      // Zero extension is the only thing we traverse that allows the size to
      // shrink if we go backwards. We have to detect this case, because if
      // something is the result of zero extension of boolean value it cannot be
      // an address for sure.
      auto *Operand = AddrArithmeticInst->getOperand(0);
      if (Operand->getType()->getIntegerBitWidth() == 1)
        rc_return IRArithmetic::unknown(AddrArithmeticInst);
    } break;

    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::BitCast:
    case Instruction::Freeze: {
      // casts are traversed
      revng_log(ModelGEPLog, "Traverse cast!");
      rc_return rc_recur getIRArithmetic(AddrArithmeticInst->getOperandUse(0),
                                         PointerTypes);
    } break;

    case Instruction::Mul: {

      auto *Op0 = AddrArithmeticInst->getOperand(0);
      auto *Op0Const = dyn_cast<ConstantInt>(Op0);
      auto *Op1 = AddrArithmeticInst->getOperand(1);
      auto *Op1Const = dyn_cast<ConstantInt>(Op1);
      auto *ConstOp = Op1Const ? Op1Const : Op0Const;
      auto *OtherOp = Op1Const ? Op0 : Op1;

      if (ConstOp and ConstOp->getValue().isNonNegative()) {
        // The constant operand is the coefficient, while the other is the
        // index.
        revng_assert(not ConstOp->isNegative());
        rc_return IRArithmetic::index(ConstOp, OtherOp);

      } else {
        // In all the other cases, fall back to treating this as a
        // non-address and non-strided instruction, just like e.g. division.
        rc_return IRArithmetic::unknown(AddrArithmeticInst);
      }

    } break;

    case Instruction::Shl: {

      auto *ShiftedBits = AddrArithmeticInst->getOperand(1);
      if (auto *ConstShift = dyn_cast<ConstantInt>(ShiftedBits)) {

        if (ConstShift->getValue().isNonNegative()) {
          // Build the stride
          auto *AddrType = AddrArithmeticInst->getType();
          auto *ArithTy = cast<llvm::IntegerType>(AddrType);
          auto *Stride = ConstantInt::get(ArithTy,
                                          1ULL << ConstShift->getZExtValue());
          if (not Stride->isNegative()) {
            // The first operand of the shift is the index
            auto *IndexForStridedAccess = AddrArithmeticInst->getOperand(0);

            rc_return IRArithmetic::index(Stride, IndexForStridedAccess);
            break;
          }
        }
      }

      // In all the other cases, fall back to treating this as a non-address
      // and non-strided instruction, just like e.g. division.

      rc_return IRArithmetic::unknown(AddrArithmeticInst);

    } break;

    case Instruction::Alloca: {
      rc_return std::nullopt;
    } break;

    case Instruction::GetElementPtr: {
      revng_abort("TODO: gep is not supported by make-model-gep yet");
    } break;

    case Instruction::Load:
    case Instruction::Call:
    case Instruction::PHI:
    case Instruction::Trunc:
    case Instruction::Select:
    case Instruction::SExt:
    case Instruction::Sub:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem: {
      // If we reach one of these instructions, it definitely cannot be an
      // address, but it's just considered as regular offset arithmetic of
      // an unknown offset.
      rc_return IRArithmetic::unknown(AddrArithmeticInst);
    } break;

    default: {
      revng_abort("Unexpected instruction for address arithmetic");
    } break;
    }

    if (ConstExprAddrArith)
      AddrArithmeticInst->deleteValue();

  } else if (auto *Const = dyn_cast<ConstantInt>(AddressArith)) {

    // If we reach this point the constant int does not represent a pointer
    // so we initialize the result as if it was an offset
    if (Const->getValue().isNonNegative())
      rc_return IRArithmetic::constant(Const);

  } else if (auto *Arg = dyn_cast<llvm::Argument>(AddressArith)) {

    // If we reach this point the argument does not represent a pointer so
    // we initialize the result as if it was an offset
    rc_return IRArithmetic::unknown(Arg);

  } else if (isa<llvm::GlobalVariable>(AddressArith)
             or isa<llvm::UndefValue>(AddressArith)
             or isa<llvm::PoisonValue>(AddressArith)) {

    rc_return std::nullopt;

  } else {
    // We don't expect other stuff. This abort is mainly intended to be a
    // safety net during development. It can eventually be dropped.
    AddressArith->dump();
    revng_abort();
  }

  rc_return std::nullopt;
}

enum MismatchScore {
  PerfectTypeMatch = 0,
  SameSize = 1,
  InRange = 2,
  OutOfBound = 3,
};

static std::string print(const MismatchScore &Match) {
  switch (Match) {
  case PerfectTypeMatch:
    return "PerfectTypeMatch";
  case SameSize:
    return "SameSize";
  case InRange:
    return "InRange";
  case OutOfBound:
    return "OutOfBound";
  default:
    return "Invalid";
  }
  return "Invalid";
}

class DifferenceScore {
  // The lower the mismatch score, the better.
  MismatchScore Mismatch = PerfectTypeMatch;

  // Higher Difference are for stuff that is farther apart from a perfect match
  // from the beginning of the type.
  IRSummation UnmatchedIR = {};

  // This field represents how deep the type system was traversed to compute the
  // score. Scores with a lower depth are considered better (so lower
  // difference).
  uint64_t Depth = std::numeric_limits<uint64_t>::min();

private:
  // Force to use factories
  DifferenceScore() = default;

  DifferenceScore(MismatchScore M, const IRSummation &Unmatched, uint64_t D) :
    Mismatch(M), UnmatchedIR(Unmatched), Depth(D) {}

private:
  // Comparison operations among IRSummation that sorts before things that are
  // "nicer" e.g they should be selected first for ModelGEP.
  std::strong_ordering
  compare(const IRSummation &LHS, const IRSummation &RHS) const {
    revng_assert(LHS.isValid() and RHS.isValid());

    // If the summations have different length, the shorter summation is
    // considered lower.
    if (auto Cmp = LHS.getIndices().size() <=> RHS.getIndices().size();
        Cmp != 0)
      return Cmp;

    // Lexicographical compare the summations.
    // The one which is lexicographically lower is considered lower.
    for (const auto &[LHSAddend, RHSAddend] :
         llvm::zip_equal(LHS.getIndices(), RHS.getIndices())) {
      if (hasCoefficientULT(LHSAddend, RHSAddend))
        return std::strong_ordering::less;
      if (hasCoefficientUGT(LHSAddend, RHSAddend))
        return std::strong_ordering::greater;
    }

    // If the two summations are equal, the one with lower Constant is lower.
    if (LHS.getConstant().ult(RHS.getConstant()))
      return std::strong_ordering::less;
    if (LHS.getConstant().ugt(RHS.getConstant()))
      return std::strong_ordering::greater;

    return std::strong_ordering::equal;
  }

public:
  std::strong_ordering operator<=>(const DifferenceScore &Other) const {

    // The one that is a better match has a lower difference.
    if (auto Cmp = Mismatch <=> Other.Mismatch; Cmp != 0)
      return Cmp;

    // Here both are have the same type of match.

    // If the Difference is not the same, one of the two has a lower difference,
    // so we give precedence to that.
    if (auto Cmp = compare(UnmatchedIR, Other.UnmatchedIR); Cmp != 0)
      return Cmp;

    // Here both have the same difference, e.g. they reach the same offset.

    // At this point we consider the shallowest (i.e. lower Depth) to be the
    // best.
    return Depth <=> Other.Depth;
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "DifferenceScore { Mismatch = " << print(Mismatch)
       << "\nUnmatchedIR = ";
    UnmatchedIR.dump(OS);
    OS << "\nDepth = " << Depth << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }

  //
  // Factories
  //

  static DifferenceScore min() { return DifferenceScore(); }

  static DifferenceScore perfectMatch(uint64_t Depth) {
    return DifferenceScore(PerfectTypeMatch, IRSummation(), Depth);
  }

  static DifferenceScore sameSizeMatch(uint64_t Depth) {
    return DifferenceScore(SameSize, IRSummation(), Depth);
  }

  static DifferenceScore inRange(uint64_t Depth) {
    return DifferenceScore(InRange, IRSummation(), Depth);
  }

  static DifferenceScore
  nestedOutOfBound(const IRSummation &DiffScore, uint64_t Depth) {
    return DifferenceScore(OutOfBound, DiffScore, Depth);
  }
};

enum AggregateKind {
  Invalid,
  Struct,
  Union,
  Array
};

static std::string toString(AggregateKind K) {
  switch (K) {
  case Struct:
    return "Struct";
  case Union:
    return "Union";
  case Array:
    return "Array";
  default:
    return "Invalid";
  }
}

// This struct represents the information necessary for building an index of a
// ModelGEP. InductionVariable can be null.
struct ChildInfo {
  IRSummation Index = IRSummation();
  AggregateKind Type = AggregateKind::Invalid;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ChildInfo{\nIndex: ";
    Index.dump(OS);
    OS << "\nType: " << toString(Type) << "\n}";
  }

  bool verify() const {
    if (Type == AggregateKind::Invalid)
      return false;
    if (Type == AggregateKind::Array)
      return true;
    return Index.isConstant();
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using ChildIndexVector = SmallVector<ChildInfo>;

// A struct that represents all the information that are necessary to replace a
// use of an llvm::Value with a combination of ModelGEP + raw pointer
// arithmetic.
// Ideally the additional raw pointer arithmetic would not be necessary, but in
// practice we can't get rid of it because the underlying type system may be
// very poor, resulting in situations where not all the pointer arithmetic in
// LLVM IR can be neatly translated in a ModelGEP.
struct ModelGEPReplacementInfo {
  model::QualifiedType BaseType;
  ChildIndexVector IndexVector;
  IRSummation Mismatched;
  model::QualifiedType AccessedType;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelGEPReplacementInfo {\nBaseType:\n";
    serialize(OS, BaseType);
    OS << "\nIndexVector: {";
    for (const auto &C : IndexVector) {
      OS << "\n";
      C.dump(OS);
    }
    OS << "}\nMismatched: {";
    Mismatched.dump(OS);
    OS << "}\nAccessedType: ";
    serialize(OS, AccessedType);
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }

  // Delete the default constructor. We want this to only be constructed with
  // valid values.
  ModelGEPReplacementInfo() = delete;

  ~ModelGEPReplacementInfo() = default;

  ModelGEPReplacementInfo(const ModelGEPReplacementInfo &) = default;
  ModelGEPReplacementInfo &operator=(const ModelGEPReplacementInfo &) = default;

  ModelGEPReplacementInfo(ModelGEPReplacementInfo &&) = default;
  ModelGEPReplacementInfo &operator=(ModelGEPReplacementInfo &&) = default;

  ModelGEPReplacementInfo(const model::QualifiedType &Base,
                          const ChildIndexVector &Indices,
                          const IRSummation &Mismatch,
                          const model::QualifiedType &Accessed) :
    BaseType(Base),
    IndexVector(Indices),
    Mismatched(Mismatch),
    AccessedType(Accessed) {}
};

using QualifiedTypeOrNone = std::optional<model::QualifiedType>;

// Compute the lower bound of DifferenceScore if we can try to match more things
// in the GEPReplacementInfo
static DifferenceScore lowerBound(const ModelGEPReplacementInfo &GEPInfo,
                                  const QualifiedTypeOrNone &AccessedTypeOnIR,
                                  model::VerifyHelper &VH) {

  const auto &[BaseOffset, Indices] = GEPInfo.Mismatched;

  size_t BaseSize = *GEPInfo.BaseType.size(VH);
  revng_assert(BaseSize);

  bool IRHasPointee = AccessedTypeOnIR.has_value();
  uint64_t AccessedSizeOnIR = 0ULL;
  if (IRHasPointee) {
    std::optional<uint64_t> AccessedSize = AccessedTypeOnIR.value().trySize(VH);
    AccessedSizeOnIR = AccessedSize.value_or(0ULL);
  }

  // The depth of a match or mismatch will be at least long as the number of
  // indices we have available.
  size_t MatchDepth = GEPInfo.IndexVector.size();

  // If the IRSum, along with the AccessedSizeOnIR is larger than the BaseType
  // it means that all the accesses in the BaseSize based on the IRSum will go
  // out ouf bound, at least with MatchDepth equal to now.
  // We cannot foresee how big will be the Mismatched part.
  // But we're estimating a lower bound, so we know that it will be out of bound
  // for at least BaseOffset + AccessedSizeOnIR - BaseSize
  if ((BaseOffset + AccessedSizeOnIR).ugt(BaseSize))
    return DifferenceScore::nestedOutOfBound(IRSummation(BaseOffset
                                                         + AccessedSizeOnIR
                                                         - BaseSize),
                                             MatchDepth);

  // In all the other cases the best outcome is a match.

  // Given that we have the IRSum to consume to reach a match, we have to assume
  // that all of it will be consumed. So we have to add to the already available
  // MatchDepth, at least 1 for each distinct coefficient Indices, or at least
  // one if we don't have any variable access.
  if (not Indices.empty())
    MatchDepth += Indices.size();
  else
    MatchDepth += BaseOffset.isZero() ? 0 : 1;

  // If the IR doesn't have a pointee type the best case scenario is an inRange.
  // We cannot have a perfectMatch or a sameSizeMatch if we don't have another
  // type to compare it with.
  if (not IRHasPointee)
    return DifferenceScore::inRange(MatchDepth);

  // If the IR has a pointee type, the best that can happen we find a perfect
  // match.
  return DifferenceScore::perfectMatch(MatchDepth);
}

// This function computes, given a BaseType, the difference score between an
// access represented on the IR (IRSum + AccessedTypeOnIR) and a potential
// ModelGEP replacemente, represented by ModelGEPInfo.
// The smaller the difference, the best ModelGEPInfo represents the memory
// access as it is computed on the IR.
static DifferenceScore
difference(const ModelGEPReplacementInfo &GEPInfo,
           const std::optional<model::QualifiedType> &AccessedTypeOnIR,
           model::VerifyHelper &VH) {

  // The number of valid indices in IndexVector represents the depth we've
  // reached so far without going out of bounds.
  auto Depth = GEPInfo.IndexVector.size();

  // If there is a non-zero mismatch, the GEPInfo represents an access that is
  // out of bound w.r.t. to the model types, at the given Depth.
  if (not GEPInfo.Mismatched.isZero())
    return DifferenceScore::nestedOutOfBound(GEPInfo.Mismatched, Depth);

  // Here Mismatch is zero

  // If the GEPInfo has an AccessedType, given that the Mismatch is zero, it
  // means that it has reached exactly the beginning of some type on the model.
  // At this point there are the following scenarios.

  // 1. If the AccessedTypeOnIR is NOT known we assume its not the same as
  // GEPInfo.AccessedType BUT we assume it's not out of bound so we return
  // inRange
  if (not AccessedTypeOnIR.has_value())
    return DifferenceScore::inRange(Depth);

  const model::QualifiedType &AccessedOnIR = AccessedTypeOnIR.value();
  const model::QualifiedType &AccessedFromModelGEP = GEPInfo.AccessedType;

  // 2. if the AccessedTypeOnIR is known, and is the same as the
  // GEPInfo.AccessedType then we have perfectMatch
  if (AccessedOnIR == AccessedFromModelGEP)
    return DifferenceScore::perfectMatch(Depth);

  std::optional<uint64_t> SizeOnIROrNOne = AccessedOnIR.trySize(VH);
  auto IRSize = SizeOnIROrNOne.value_or(0ULL);
  auto GEPSize = *AccessedFromModelGEP.size(VH);
  revng_assert(GEPSize);
  auto Cmp = IRSize <=> GEPSize;

  // 3. if the AccessedTypeOnIR is known, and is NOT the same as the
  // GEPInfo.AccessedType BUT it has the same size then we have sameSizeMatch
  if (Cmp == 0)
    return DifferenceScore::sameSizeMatch(Depth);

  // 4. if the AccessedTypeOnIR is known, and is NOT the same as the
  // GEPInfo.AccessedType AND it has a smaller size then we have inRange
  if (Cmp < 0)
    return DifferenceScore::inRange(Depth);

  // 5. if the AccessedTypeOnIR is known, and is NOT the same as the
  // GEPInfo.AccessedType AND it has a larger size then we have an out of bound
  return DifferenceScore::nestedOutOfBound(IRSummation(IRSize), Depth);
}

static RecursiveCoroutine<ModelGEPReplacementInfo>
computeBest(const model::QualifiedType &BaseType,
            const IRSummation &IRSum,
            const std::optional<model::QualifiedType> &AccessedTypeOnIR,
            model::VerifyHelper &VH);

static RecursiveCoroutine<ModelGEPReplacementInfo>
computeBestInArray(const model::QualifiedType &BaseType,
                   const IRSummation &IRSum,
                   const std::optional<model::QualifiedType> &AccessedTypeOnIR,
                   model::VerifyHelper &VH) {

  revng_log(ModelGEPLog, "computeBestInArray for IRSum: " << IRSum);
  auto ArrayIndent = LoggerIndent{ ModelGEPLog };

  model::QualifiedType BaseArray = peelConstAndTypedefs(BaseType);
  revng_assert(BaseArray.isArray());

  // In case of arrays, we need to unwrap the first layer
  // (qualifier) and keep looking for other TAPs that might be generated.
  auto QBeg = BaseArray.Qualifiers().begin();
  auto QEnd = BaseArray.Qualifiers().end();

  auto &ArrayQualifier = *QBeg;
  revng_assert(model::Qualifier::isArray(ArrayQualifier));
  auto ElementType = model::QualifiedType(BaseArray.UnqualifiedType(),
                                          { std::next(QBeg), QEnd });

  std::optional<uint64_t> MaybeElementTypeSize = ElementType.size(VH);
  revng_assert(MaybeElementTypeSize.has_value()
               and MaybeElementTypeSize.value());
  uint64_t ElementSize = MaybeElementTypeSize.value();
  uint64_t NElems = ArrayQualifier.Size();
  uint64_t ArraySize = ElementSize * NElems;

  const auto &[BaseOffset, Indices] = IRSum;

  // Setup a return value for all the cases where we cannot traverse the array
  // successfully. We basically emit no index and keep the whol IRSum as
  // Mismatch.
  ModelGEPReplacementInfo ArrayNotTraversed(BaseArray, {}, IRSum, BaseArray);

  bool IRHasPointee = AccessedTypeOnIR.has_value();
  uint64_t AccessedSizeOnIR = 0ULL;
  if (IRHasPointee) {
    std::optional<uint64_t> AccessedSize = AccessedTypeOnIR.value().trySize(VH);
    AccessedSizeOnIR = AccessedSize.value_or(0ULL);
  }

  if ((BaseOffset + AccessedSizeOnIR).ugt(ArraySize)) {
    revng_log(ModelGEPLog,
              "Array not traversed for large offset. BestInArray: "
                << ArrayNotTraversed);
    rc_return ArrayNotTraversed;
  }

  APInt APElementSize = APInt(/*NumBits*/ 64, /*value*/ ElementSize);

  // Now, if Indices is not empty, we also have to compute the variable part of
  // the array access.
  std::optional<SmallVector<IRAddend>> InductionVariable = std::nullopt;
  SmallVector<IRAddend> InnerIndices = Indices;
  auto It = InnerIndices.begin();
  // Find the first element in InnerIndices that has a Coefficient that is
  // smaller than APElementSize.
  auto End = llvm::upper_bound(InnerIndices,
                               APElementSize,
                               [](const APInt &LHS, const IRAddend &RHS) {
                                 APInt RHSValue = RHS.coefficient()->getValue();
                                 return coefficientUGT(LHS, RHSValue);
                               });
  for (const auto &InnerIndex : llvm::make_range(It, End)) {

    revng_log(ModelGEPLog, "Trying to traverse array index: " << InnerIndex);
    const auto &[Coefficient, Index] = InnerIndex;
    revng_assert(not isa<Constant>(Index));

    // Unwrap the top layer of array indices, since we're traversing the
    // array. In principle, we should unwrap it only if the Coefficient
    // matches the size of the element of the array. In practice, this is
    // overly restrictive, because it doesn't allow us to represent array
    // accesses that access e.g. all the even elements in an array (for which
    // the Coefficient would be a multiple of the element size).
    //
    // So we attempt to unwrap the top layer of array indices in all the cases
    // where Coefficient is larger than the element size.
    // At this point there are 2 scenarios.
    // - the array access is done with a Coefficient that is a multiple of the
    // element size: these are good and we want to handle them.
    // - the array access is done with a Coefficient that is NOT a multiple of
    // the element size: these are bad and we want to discard them.
    //
    // In all the cases where we can successfully unwrap, the non-constant
    // Index is the induction variable and we have to track it.
    //
    // In all the other cases, where the Coefficient is lower than the element
    // size, we don't unwrap, because the unwrap will occur in a more nested
    // level, if necessary.
    if (Coefficient->getValue().zext(64).ugt(APElementSize)) {
      revng_log(ModelGEPLog, "Large coefficient");
      revng_assert(Index);

      APInt NumMultipleElements = APInt(/*NumBits*/ 64, /*value*/ 0);
      APInt Remainder = APInt(/*NumBits*/ 64, /*value*/ 0);

      APInt::udivrem(Coefficient->getValue().zext(64),
                     APElementSize,
                     NumMultipleElements,
                     Remainder);
      if (Remainder.getBoolValue()) {

        // This is not accessing an exact size of elements at each stride.
        // Just skip this.
        revng_log(ModelGEPLog,
                  "Array not traversed for large coefficient. BestInArray: "
                    << ArrayNotTraversed);
        rc_return ArrayNotTraversed;
      } else {

        revng_log(ModelGEPLog, "Coefficient is multiple of element size.");
        if (not InductionVariable.has_value())
          InductionVariable = SmallVector<IRAddend>{};

        auto *Int64Type = llvm::IntegerType::getIntNTy(Index->getContext(), 64);
        IRAddend NewAddend = IRAddend{
          cast<ConstantInt>(ConstantInt::get(Int64Type, NumMultipleElements)),
          Index
        };
        InductionVariable->push_back(std::move(NewAddend));
      }
    } else if (Coefficient->getValue() == ElementSize) {

      revng_log(ModelGEPLog, "Coefficient is exact");
      if (not InductionVariable.has_value())
        InductionVariable = SmallVector<IRAddend>{};

      InductionVariable->push_back(IRAddend(Index));
    } else {
      revng_abort();
    }
  }
  InnerIndices.erase(It, End);

  // Here BaseOffset might be larger than the ElementSize.
  // Compute the minimum number of elements that will be skipped by pointer
  // arithmetic, and also the new BaseOffset within the accessed element.
  APInt BaseOffsetInElement;
  APInt FixedElementIndex;
  APInt::udivrem(BaseOffset,
                 APElementSize,
                 FixedElementIndex,
                 BaseOffsetInElement);

  auto SummationInElement = IRSummation(BaseOffsetInElement,
                                        std::move(InnerIndices));
  IRSummation ElementSummation = IRSummation(FixedElementIndex);
  if (InductionVariable.has_value())
    ElementSummation += IRSummation(std::move(InductionVariable.value()));
  auto ArrayAccessInfo = ChildInfo{ .Index = std::move(ElementSummation),
                                    .Type = AggregateKind::Array };

  ModelGEPReplacementInfo BestInArray(BaseArray,
                                      { ArrayAccessInfo },
                                      SummationInElement,
                                      ElementType);

  revng_log(ModelGEPLog, "BestInArray: " << BestInArray);
  DifferenceScore BestScore = difference(BestInArray, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "BestInArray Score: " << BestScore);

  DifferenceScore LowerBound = lowerBound(BestInArray, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "lowerBound: " << LowerBound);
  if (LowerBound >= BestScore) {
    revng_log(ModelGEPLog, "Cannot improve: bail out");
    rc_return BestInArray;
  }

  revng_log(ModelGEPLog, "Analyze array element");
  auto ElementIndex = LoggerIndent{ ModelGEPLog };

  auto ElementResult = rc_recur computeBest(ElementType,
                                            SummationInElement,
                                            AccessedTypeOnIR,
                                            VH);
  // Fixup the ElementResult to be comparable with BestInArray
  ElementResult.BaseType = BaseArray;
  ElementResult.IndexVector.insert(ElementResult.IndexVector.begin(),
                                   ArrayAccessInfo);

  revng_log(ModelGEPLog, "ElementResult: " << ElementResult);

  DifferenceScore ElementScore = difference(ElementResult,
                                            AccessedTypeOnIR,
                                            VH);
  revng_log(ModelGEPLog, "ElementResult Score: " << ElementScore);

  if (ElementScore < BestScore) {
    revng_log(ModelGEPLog, "New BestInArray: " << ElementResult);
    BestInArray = ElementResult;
  }

  rc_return BestInArray;
}

static RecursiveCoroutine<ModelGEPReplacementInfo>
computeBestInStruct(const model::QualifiedType &BaseStruct,
                    const IRSummation &IRSum,
                    const std::optional<model::QualifiedType> &AccessedTypeOnIR,
                    model::VerifyHelper &VH) {
  revng_log(ModelGEPLog, "computeBestInStruct for IRSum: " << IRSum);
  auto StructIndent = LoggerIndent{ ModelGEPLog };

  revng_assert(BaseStruct.is(model::TypeKind::StructType)
               and BaseStruct.Qualifiers().empty());
  const auto *Unqualified = BaseStruct.UnqualifiedType().getConst();
  const auto *TheStructType = cast<model::StructType>(Unqualified);

  // Setup a return value for all the cases where we cannot traverse the union
  // successfully. We basically emit no index and keep the whol IRSum as
  // Mismatch.
  ModelGEPReplacementInfo BestInStruct(BaseStruct, {}, IRSum, BaseStruct);

  bool IRHasPointee = AccessedTypeOnIR.has_value();
  uint64_t AccessedSizeOnIR = 0ULL;
  if (IRHasPointee) {
    std::optional<uint64_t> AccessedSize = AccessedTypeOnIR.value().trySize(VH);
    AccessedSizeOnIR = AccessedSize.value_or(0ULL);
  }

  const auto &[BaseOffset, Indices] = IRSum;

  auto StructSize = *BaseStruct.size(VH);
  if ((BaseOffset + AccessedSizeOnIR).ugt(StructSize)) {
    revng_log(ModelGEPLog,
              "Struct not traversed for large offset. BestInStruct: "
                << BestInStruct);
    rc_return BestInStruct;
  }

  revng_log(ModelGEPLog, "BestInStruct: " << BestInStruct);
  DifferenceScore BestScore = difference(BestInStruct, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "BestInStruct Score: " << BestScore);

  DifferenceScore LowerBound = lowerBound(BestInStruct, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "lowerBound: " << LowerBound);
  if (LowerBound >= BestScore) {
    revng_log(ModelGEPLog, "Cannot improve: bail out");
    rc_return BestInStruct;
  }

  // Now we try to unwrap the union fields, and see if we can get better results
  // on them.

  auto FieldBegin = TheStructType->Fields().begin();
  // Let's detect the leftmost field that starts later than the maximum offset
  // reachable by the IRSum. This is the first element that we don't want to
  // compare because it's not a valid traversal for the given IRSum.
  auto FieldIt = TheStructType->Fields().upper_bound(BaseOffset.getZExtValue());

  enum {
    InitiallyImproving,
    StartedDegrading,
  } Status = InitiallyImproving;

  for (const model::StructField &Field :
       llvm::reverse(llvm::make_range(FieldBegin, FieldIt))) {
    revng_log(ModelGEPLog, "Analyze Field with Offset: " << Field.Offset());
    auto FieldIndent = LoggerIndent{ ModelGEPLog };

    // At this point we assume we've accessed at least one field.
    auto FieldAccessInfo = ChildInfo{ .Index = IRSummation(Field.Offset()),
                                      .Type = AggregateKind::Struct };

    auto &FieldType = Field.Type();

    IRSummation SumInField = IRSum;
    SumInField -= Field.Offset();
    ModelGEPReplacementInfo InField(BaseStruct,
                                    { FieldAccessInfo },
                                    SumInField,
                                    FieldType);

    DifferenceScore ElementLowerBound = lowerBound(InField,
                                                   AccessedTypeOnIR,
                                                   VH);
    revng_log(ModelGEPLog, "lowerBound: " << LowerBound);

    // If we found a lower bound that is better, it should imply that we're
    // still in the initial phase of improvement.
    revng_assert(LowerBound >= BestScore or Status == InitiallyImproving);

    if (LowerBound >= BestScore) {
      revng_log(ModelGEPLog, "Cannot improve on this Field");
      Status = StartedDegrading;
      continue;
    }

    auto FieldResult = rc_recur computeBest(FieldType,
                                            SumInField,
                                            AccessedTypeOnIR,
                                            VH);
    // Fixup the FieldResult to be comparable with BestInStruct
    FieldResult.BaseType = BaseStruct;
    FieldResult.IndexVector.insert(FieldResult.IndexVector.begin(),
                                   FieldAccessInfo);

    revng_log(ModelGEPLog, "FieldResult: " << FieldResult);

    DifferenceScore FieldScore = difference(FieldResult, AccessedTypeOnIR, VH);
    revng_log(ModelGEPLog, "FieldResult Score: " << FieldScore);

    if (FieldScore < BestScore) {
      BestInStruct = FieldResult;
      BestScore = FieldScore;
      revng_log(ModelGEPLog, "New BestInStruct: " << BestInStruct);
    }
  }

  rc_return BestInStruct;
}

static RecursiveCoroutine<ModelGEPReplacementInfo>
computeBestInUnion(const model::QualifiedType &BaseUnion,
                   const IRSummation &IRSum,
                   const std::optional<model::QualifiedType> &AccessedTypeOnIR,
                   model::VerifyHelper &VH) {

  revng_log(ModelGEPLog, "computeBestInUnion for IRSum: " << IRSum);
  auto UnionIndent = LoggerIndent{ ModelGEPLog };

  revng_assert(BaseUnion.is(model::TypeKind::UnionType)
               and BaseUnion.Qualifiers().empty());
  const auto *Unqualified = BaseUnion.UnqualifiedType().getConst();
  const auto *TheUnionType = cast<model::UnionType>(Unqualified);

  // Setup a return value for all the cases where we cannot traverse the union
  // successfully. We basically emit no index and keep the whol IRSum as
  // Mismatch.
  ModelGEPReplacementInfo BestInUnion(BaseUnion, {}, IRSum, BaseUnion);

  bool IRHasPointee = AccessedTypeOnIR.has_value();
  uint64_t AccessedSizeOnIR = 0ULL;
  if (IRHasPointee) {
    std::optional<uint64_t> AccessedSize = AccessedTypeOnIR.value().trySize(VH);
    AccessedSizeOnIR = AccessedSize.value_or(0ULL);
  }

  const auto &[BaseOffset, Indices] = IRSum;

  auto UnionSize = *BaseUnion.size(VH);
  if ((BaseOffset + AccessedSizeOnIR).ugt(UnionSize)) {
    revng_log(ModelGEPLog,
              "Union not traversed for large offset. BestInUnion: "
                << BestInUnion);
    rc_return BestInUnion;
  }

  revng_log(ModelGEPLog, "BestInUnion: " << BestInUnion);
  DifferenceScore BestScore = difference(BestInUnion, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "BestInUnion Score: " << BestScore);

  DifferenceScore LowerBound = lowerBound(BestInUnion, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "lowerBound: " << LowerBound);
  if (LowerBound >= BestScore) {
    revng_log(ModelGEPLog, "Cannot improve: bail out");
    rc_return BestInUnion;
  }

  // Now we try to unwrap the union fields, and see if we can get better results
  // on them.

  for (const model::UnionField &Field : TheUnionType->Fields()) {
    revng_log(ModelGEPLog, "Analyze Field with ID: " << Field.Index());
    auto FieldIndent = LoggerIndent{ ModelGEPLog };

    // At this point we assume we've accessed at least one field.
    auto FieldAccessInfo = ChildInfo{ .Index = IRSummation(Field.Index()),
                                      .Type = AggregateKind::Union };

    auto &FieldType = Field.Type();

    ModelGEPReplacementInfo InField(BaseUnion,
                                    { FieldAccessInfo },
                                    IRSum,
                                    FieldType);

    DifferenceScore ElementLowerBound = lowerBound(InField,
                                                   AccessedTypeOnIR,
                                                   VH);
    revng_log(ModelGEPLog, "lowerBound: " << LowerBound);
    if (LowerBound >= BestScore) {
      revng_log(ModelGEPLog, "Cannot improve on this Field");
      continue;
    }

    auto FieldResult = rc_recur computeBest(FieldType,
                                            IRSum,
                                            AccessedTypeOnIR,
                                            VH);
    // Fixup the FieldResult to be comparable with BestInUnion
    FieldResult.BaseType = BaseUnion;
    FieldResult.IndexVector.insert(FieldResult.IndexVector.begin(),
                                   FieldAccessInfo);

    revng_log(ModelGEPLog, "FieldResult: " << FieldResult);

    DifferenceScore FieldScore = difference(FieldResult, AccessedTypeOnIR, VH);
    revng_log(ModelGEPLog, "FieldResult Score: " << FieldScore);

    if (FieldScore < BestScore) {
      BestInUnion = FieldResult;
      BestScore = FieldScore;
      revng_log(ModelGEPLog, "New BestInUnion: " << BestInUnion);
    }
  }

  rc_return BestInUnion;
}

static RecursiveCoroutine<ModelGEPReplacementInfo>
computeBest(const model::QualifiedType &BaseType,
            const IRSummation &IRSum,
            const std::optional<model::QualifiedType> &AccessedTypeOnIR,
            model::VerifyHelper &VH) {
  revng_log(ModelGEPLog, "Computing Best ModelGEP for IRSum: " << IRSum);
  revng_assert(not BaseType.isVoid()
               and not BaseType.is(model::TypeKind::RawFunctionType)
               and not BaseType.is(model::TypeKind::CABIFunctionType));

  // This Result models no access, and leaves all the IRSum as a mismatch.
  // It's handful for easy bail out for obvious failures to traverse this
  // BaseType properly.
  ModelGEPReplacementInfo Result{ BaseType, {}, IRSum, BaseType };

  // If we've reached a type that cannot be "traversed" by a ModelGEP we have
  // nothing to do. This might be subject to cheange in the future when we'll
  // need to support "accessing" pointer with the square brackets as if they
  // were arrays.
  if (BaseType.isPrimitive() or BaseType.isPointer()
      or BaseType.is(model::TypeKind::EnumType)) {
    revng_log(ModelGEPLog,
              "Never replace pointer arithmetic from a type that is not a "
              "struct, a union, or an array, with a ModelGEP");
    rc_return Result;
  }

  revng_log(ModelGEPLog, "Result: " << Result);
  DifferenceScore BestScore = difference(Result, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "BestScore: " << BestScore);

  DifferenceScore LowerBound = lowerBound(Result, AccessedTypeOnIR, VH);
  revng_log(ModelGEPLog, "lowerBound: " << LowerBound);

  if (LowerBound >= BestScore) {
    revng_log(ModelGEPLog, "Cannot improve: bail out");
    rc_return Result;
  }

  auto UnwrappedBaseType = peelConstAndTypedefs(BaseType);

  if (UnwrappedBaseType.isArray()) {
    revng_log(ModelGEPLog, "Array");
    ModelGEPReplacementInfo ArrayResult = rc_recur
      computeBestInArray(UnwrappedBaseType, IRSum, AccessedTypeOnIR, VH);
    revng_log(ModelGEPLog, "ArrayResult: " << ArrayResult);

    DifferenceScore ArrayBestScore = difference(ArrayResult,
                                                AccessedTypeOnIR,
                                                VH);
    revng_log(ModelGEPLog, "ArrayBestScore: " << ArrayBestScore);

    if (ArrayBestScore < BestScore) {
      Result = ArrayResult;
      revng_log(ModelGEPLog, "New BestScore: " << Result);
    }

    rc_return Result;
  }

  // TODO: we're gonna need to handle pointers in the future.
  revng_assert(UnwrappedBaseType.Qualifiers().empty());

  const model::Type *BaseT = UnwrappedBaseType.UnqualifiedType().getConst();

  switch (BaseT->Kind()) {

  case model::TypeKind::StructType: {
    Result = rc_recur computeBestInStruct(UnwrappedBaseType,
                                          IRSum,
                                          AccessedTypeOnIR,
                                          VH);
  } break;

  case model::TypeKind::UnionType: {
    Result = rc_recur computeBestInUnion(UnwrappedBaseType,
                                         IRSum,
                                         AccessedTypeOnIR,
                                         VH);
  } break;

  default:
    revng_abort();
  }

  rc_return Result;
}

static model::QualifiedType getType(const model::QualifiedType &BaseType,
                                    const ChildIndexVector &IndexVector,
                                    model::VerifyHelper &VH) {

  model::QualifiedType CurrType = BaseType;
  bool TopLevel = true;
  for (const auto &[Index, AggregateType] : IndexVector) {

    switch (AggregateType) {

    case AggregateKind::Struct: {

      revng_assert(Index.isConstant());
      CurrType = peelConstAndTypedefs(CurrType);
      auto *S = cast<model::StructType>(CurrType.UnqualifiedType().getConst());
      CurrType = S->Fields().at(Index.getConstant().getZExtValue()).Type();

    } break;

    case AggregateKind::Union: {

      revng_assert(Index.isConstant());
      CurrType = peelConstAndTypedefs(CurrType);
      auto *U = cast<model::UnionType>(CurrType.UnqualifiedType().getConst());
      CurrType = U->Fields().at(Index.getConstant().getZExtValue()).Type();

    } break;

    case AggregateKind::Array: {

      CurrType = peelConstAndTypedefs(CurrType);
      auto Begin = CurrType.Qualifiers().begin();
      revng_assert(Begin != CurrType.Qualifiers().end());

      {
        [[maybe_unused]] bool IsArray = model::Qualifier::isArray(*Begin);
        [[maybe_unused]] bool IsPointer = model::Qualifier::isPointer(*Begin);
        [[maybe_unused]] auto IndexValue = Index.getConstant().getZExtValue();
        revng_assert((IsArray and Begin->Size() > IndexValue)
                     or (IsPointer and TopLevel));
      }

      // For arrays we don't need to look at the value of the index, we just
      // unwrap the array and go on.
      CurrType = model::QualifiedType(CurrType.UnqualifiedType(),
                                      { std::next(Begin),
                                        CurrType.Qualifiers().end() });
    } break;

    default:
      revng_abort();
    }

    TopLevel = false;
  }

  return CurrType;
}

using UseTypeMap = std::map<Use *, model::QualifiedType>;

static std::optional<model::QualifiedType>
getAccessedTypeOnIR(FunctionMetadataCache &Cache,
                    const Use &U,
                    const model::Binary &Model,
                    const ModelTypesMap &PointerTypes,
                    const UseTypeMap &GEPifiedUseTypes) {

  auto *UserInstr = dyn_cast<Instruction>(U.getUser());
  if (nullptr == UserInstr)
    return std::nullopt;

  switch (UserInstr->getOpcode()) {

  case llvm::Instruction::Load: {
    auto *Load = cast<llvm::LoadInst>(UserInstr);
    revng_log(ModelGEPLog, "User is Load");
    // If the user of U is a load, we know that the pointee's size is equal to
    // the size of the loaded value
    revng_assert(Load->getType()->isIntOrPtrTy());
    const llvm::DataLayout &DL = UserInstr->getModule()->getDataLayout();
    auto PointeeSize = DL.getTypeAllocSize(Load->getType());

    model::QualifiedType QPointee;
    auto *PtrOp = Load->getPointerOperand();
    auto *PtrOpUse = &Load->getOperandUse(Load->getPointerOperandIndex());
    if (auto It = PointerTypes.find(PtrOp); It != PointerTypes.end()) {
      QPointee = dropPointer(It->second);
    }
    if (auto It = GEPifiedUseTypes.find(PtrOpUse);
        It != GEPifiedUseTypes.end()) {
      QPointee = dropPointer(It->second);
    } else {
      model::TypePath
        Pointee = Model.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                         PointeeSize);
      QPointee = model::QualifiedType(Pointee, {});
    }

    revng_log(ModelGEPLog, "QPointee: " << serializeToString(QPointee));
    return QPointee;
  } break;

  case llvm::Instruction::Store: {
    auto *Store = cast<llvm::StoreInst>(UserInstr);
    revng_log(ModelGEPLog, "User is Store");
    // If the user of U is a store, and U is the pointer operand, we know
    // that the pointee's size is equal to the size of the stored value.
    if (U.getOperandNo() == llvm::StoreInst::getPointerOperandIndex()) {
      revng_log(ModelGEPLog, "Use is pointer operand");
      auto *Stored = Store->getValueOperand();

      revng_assert(Stored->getType()->isIntOrPtrTy());
      const llvm::DataLayout &DL = UserInstr->getModule()->getDataLayout();
      unsigned long PointeeSize = DL.getTypeAllocSize(Stored->getType());

      model::TypePath
        Pointee = Model.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                         PointeeSize);
      model::QualifiedType Generic = model::QualifiedType(Pointee, {});
      model::QualifiedType QPointee = Generic;

      auto *PtrOp = Store->getPointerOperand();
      auto *PtrOpUse = &Store->getOperandUse(Store->getPointerOperandIndex());

      if (auto It = PointerTypes.find(PtrOp); It != PointerTypes.end()) {
        QPointee = dropPointer(It->second);
        if (QPointee.is(model::TypeKind::StructType)
            or QPointee.is(model::TypeKind::UnionType)) {
          QPointee = Generic;
        }
      }
      if (auto It = GEPifiedUseTypes.find(PtrOpUse);
          It != GEPifiedUseTypes.end()) {
        QPointee = dropPointer(It->second);
        if (QPointee.is(model::TypeKind::StructType)
            or QPointee.is(model::TypeKind::UnionType)) {
          QPointee = Generic;
        }
      }

      revng_log(ModelGEPLog, "QPointee: " << serializeToString(QPointee));
      return QPointee;
    } else {
      revng_log(ModelGEPLog, "Use is pointer operand");
    }

  } break;

  case llvm::Instruction::Ret: {
    auto *Ret = cast<llvm::ReturnInst>(UserInstr);
    llvm::Function *ReturningF = Ret->getFunction();

    // If the user is a ret, we want to look at the return type of the
    // function we're returning from, and use it as a pointee type.

    const model::Function *MF = llvmToModelFunction(Model, *ReturningF);
    revng_assert(MF);

    const auto Layout = abi::FunctionType::Layout::make(MF->Prototype());

    bool HasNoReturnValues = (Layout.ReturnValues.empty()
                              and not Layout.returnsAggregateType());
    const model::QualifiedType *SingleReturnType = nullptr;

    if (Layout.returnsAggregateType()) {
      SingleReturnType = &Layout.Arguments[0].Type;
    } else if (Layout.ReturnValues.size() == 1) {
      SingleReturnType = &Layout.ReturnValues[0].Type;
    }

    // If the callee function does not return anything, skip to the next
    // instruction.
    if (HasNoReturnValues) {
      revng_log(ModelGEPLog, "Does not return values in the model. Skip ...");
      revng_assert(not Ret->getReturnValue());
    } else if (SingleReturnType != nullptr) {
      revng_log(ModelGEPLog, "Has a single return value.");

      revng_assert(Ret->getReturnValue()->getType()->isVoidTy()
                   or Ret->getReturnValue()->getType()->isIntOrPtrTy());

      // If the returned type is a pointer, we unwrap it and set the pointee
      // type of IRPattern to the pointee of the return type.
      // Otherwise the Function is not returning a pointer, and we can skip
      // it.
      if (SingleReturnType->isPointer()) {
        auto _ = LoggerIndent(ModelGEPLog);
        revng_log(ModelGEPLog, "llvm::ReturnInst: " << dumpToString(Ret));
        revng_log(ModelGEPLog,
                  "Pointee: model::QualifiedType: "
                    << serializeToString(*SingleReturnType));
        return dropPointer(*SingleReturnType);
      }

    } else {
      auto *RetVal = Ret->getReturnValue();
      auto *StructTy = cast<llvm::StructType>(RetVal->getType());
      revng_log(ModelGEPLog, "Has many return types.");
      auto ReturnValuesCount = Layout.ReturnValues.size();
      revng_assert(StructTy->getNumElements() == ReturnValuesCount);

      // Assert that we're returning a proper struct, initialized with
      // struct initializers, but don't do anything here.
      const auto *Returned = cast<llvm::CallInst>(RetVal)->getCalledFunction();
      revng_assert(FunctionTags::StructInitializer.isTagOf(Returned));
    }

  } break;

  case llvm::Instruction::Call: {
    auto *Call = dyn_cast<llvm::CallInst>(UserInstr);
    // If the user is a call, and it's calling an isolated function we want to
    // look at the argument types of the callee on the model, and use info
    // coming from them for initializing IRPattern.AccessedType

    revng_log(ModelGEPLog, "Call");

    if (isCallToTagged(Call, FunctionTags::StructInitializer)) {

      const llvm::Function *CalledF = Call->getCalledFunction();

      // special case for struct initializers
      unsigned ArgNum = Call->getArgOperandNo(&U);

      const model::Function *CalledFType = llvmToModelFunction(Model, *CalledF);
      const model::Type *CalledPrototype = CalledFType->Prototype().getConst();

      if (auto *RFT = dyn_cast<RawFunctionType>(CalledPrototype)) {
        revng_log(ModelGEPLog, "Has RawFunctionType prototype.");
        revng_assert(RFT->ReturnValues().size() > 1);

        auto *StructTy = cast<llvm::StructType>(CalledF->getReturnType());
        revng_log(ModelGEPLog, "Has many return types.");
        auto ValuesCount = RFT->ReturnValues().size();
        revng_assert(StructTy->getNumElements() == ValuesCount);

        model::QualifiedType
          RetTy = std::next(RFT->ReturnValues().begin(), ArgNum)->Type();
        if (RetTy.isPointer()) {
          model::QualifiedType Pointee = dropPointer(RetTy);
          revng_log(ModelGEPLog, "Pointee: " << serializeToString(Pointee));
          return Pointee;
        }
      } else if (auto *CFT = dyn_cast<CABIFunctionType>(CalledPrototype)) {
        revng_log(ModelGEPLog, "Has CABIFunctionType prototype.");
        // TODO: we haven't handled return values of CABIFunctions yet
        revng_abort();
      } else {
        revng_abort("Function should have RawFunctionType or "
                    "CABIFunctionType");
      }

    } else if (isCallToIsolatedFunction(Call)) {
      auto Proto = Cache.getCallSitePrototype(Model, Call);
      revng_assert(Proto.isValid());

      if (const auto *RFT = dyn_cast<RawFunctionType>(Proto.get())) {

        auto MoreIndent = LoggerIndent(ModelGEPLog);
        auto ModelArgSize = RFT->Arguments().size();
        revng_assert(RFT->StackArgumentsType().Qualifiers().empty());
        auto &Type = RFT->StackArgumentsType().UnqualifiedType();
        revng_assert((ModelArgSize == Call->arg_size() - 1 and Type.isValid())
                     or ModelArgSize == Call->arg_size());
        revng_log(ModelGEPLog, "model::RawFunctionType");

        auto _ = LoggerIndent(ModelGEPLog);
        if (not Call->isCallee(&U)) {
          unsigned ArgOpNum = Call->getArgOperandNo(&U);
          revng_log(ModelGEPLog, "ArgOpNum: " << ArgOpNum);
          revng_log(ModelGEPLog, "ArgOperand: " << U.get());

          model::QualifiedType ArgTy;
          if (ArgOpNum >= ModelArgSize) {
            // The only case in which the argument's index can be greater than
            // the number of arguments in the model is for RawFunctionType
            // functions that have stack arguments.
            // Stack arguments are passed as the last argument of the llvm
            // function, but they do not have a corresponding argument in the
            // model. In this case, we have to retrieve the StackArgumentsType
            // from the function prototype.
            ArgTy = RFT->StackArgumentsType();
            revng_assert(ArgTy.UnqualifiedType().isValid());
          } else {
            auto ArgIt = std::next(RFT->Arguments().begin(), ArgOpNum);
            ArgTy = ArgIt->Type();
          }

          revng_log(ModelGEPLog,
                    "model::QualifiedType: " << serializeToString(ArgTy));
          if (ArgTy.isPointer()) {
            model::QualifiedType Pointee = dropPointer(ArgTy);
            revng_log(ModelGEPLog, "Pointee: " << serializeToString(Pointee));
            return Pointee;
          }
        } else {
          revng_log(ModelGEPLog, "IsCallee");
        }

      } else if (const auto *CFT = dyn_cast<CABIFunctionType>(Proto.get())) {

        auto MoreIndent = LoggerIndent(ModelGEPLog);
        revng_assert(CFT->Arguments().size() == Call->arg_size());
        revng_log(ModelGEPLog, "model::CABIFunctionType");

        auto _ = LoggerIndent(ModelGEPLog);
        if (not Call->isCallee(&U)) {
          unsigned ArgOpNum = Call->getArgOperandNo(&U);
          revng_log(ModelGEPLog, "ArgOpNum: " << ArgOpNum);
          revng_log(ModelGEPLog, "ArgOperand: " << U.get());
          model::QualifiedType ArgTy = CFT->Arguments().at(ArgOpNum).Type();
          revng_log(ModelGEPLog,
                    "model::QualifiedType: " << serializeToString(ArgTy));
          if (ArgTy.isPointer()) {
            model::QualifiedType Pointee = dropPointer(ArgTy);
            revng_log(ModelGEPLog, "Pointee: " << serializeToString(Pointee));
            return Pointee;
          }
        } else {
          revng_log(ModelGEPLog, "IsCallee");
        }

      } else {
        revng_abort("Function should have RawFunctionType or "
                    "CABIFunctionType");
      }
    }

  } break;

  default:
    return std::nullopt;
  }

  return std::nullopt;
}

struct UseReplacementWithModelGEP {
  Use *U;
  Value *BaseAddress;
  ModelGEPReplacementInfo ReplacementInfo;
};

static std::vector<UseReplacementWithModelGEP>
makeGEPReplacements(llvm::Function &F,
                    const model::Binary &Model,
                    model::VerifyHelper &VH,
                    FunctionMetadataCache &Cache) {

  std::vector<UseReplacementWithModelGEP> Result;

  const model::Function *ModelF = llvmToModelFunction(Model, F);
  revng_assert(ModelF);

  // First, try to initialize a map for the known model types of llvm::Values
  // that are reachable from F. If this fails, we just bail out because we
  // cannot infer any modelGEP in F, if we have no type information to rely
  // on.
  ModelTypesMap PointerTypes = initModelTypes(Cache,
                                              F,
                                              ModelF,
                                              Model,
                                              /*PointersOnly=*/true);
  if (PointerTypes.empty()) {
    revng_log(ModelGEPLog, "Model Types not found for " << F.getName());
    return Result;
  }

  UseTypeMap GEPifiedUsedTypes;

  auto RPOT = ReversePostOrderTraversal(&F.getEntryBlock());
  for (auto *BB : RPOT) {
    for (auto &I : *BB) {
      revng_log(ModelGEPLog, "Instruction " << dumpToString(&I));
      auto Indent = LoggerIndent{ ModelGEPLog };

      if (auto *CallI = dyn_cast<llvm::CallInst>(&I)) {
        if (not isCallToIsolatedFunction(CallI)) {
          revng_log(ModelGEPLog, "Skipping call to non-isolated function");
          continue;
        }
      }

      for (Use &U : I.operands()) {

        // Skip everything that is not a pointer or an integer, since it
        // cannot be an any kind of pointer arithmetic that we handle
        if (not U.get()->getType()->isIntOrPtrTy()) {
          revng_log(ModelGEPLog,
                    "Skipping operand that cannot be pointer arithmetic");
          continue;
        }

        // Skip non-pointer-sized integers, since they cannot be addresses
        if (auto *IntTy = dyn_cast<llvm::IntegerType>(U.get()->getType())) {
          using model::Architecture::getPointerSize;
          auto PtrBitSize = getPointerSize(Model.Architecture()) * 8;
          if (IntTy->getIntegerBitWidth() != PtrBitSize) {
            revng_log(ModelGEPLog, "Skipping i1 value");
            continue;
          }
        }

        // Skip callee operands in CallInst if it's not an llvm::Instruction.
        // If it is an Instruction we should handle it.
        if (auto *CallUser = dyn_cast<llvm::CallInst>(U.getUser())) {
          if (not isa<llvm::Instruction>(CallUser->getCalledOperand())) {
            if (&U == &CallUser->getCalledOperandUse()) {
              revng_log(ModelGEPLog, "Skipping callee operand in CallInst");
              continue;
            }
          }
        }

        // Skip null pointer constants, and undefs, since they cannot be valid
        // addresses
        // TODO: if we ever need to support memory mapped at address 0 we can
        // probably work around this, but this is not top priority for now.
        if (isa<llvm::UndefValue>(U.get()) or isa<llvm::PoisonValue>(U.get())
            or isa<llvm::ConstantPointerNull>(U.get())) {
          revng_log(ModelGEPLog, "Skipping null pointer address");
          continue;
        }

        std::optional<IRArithmetic>
          IRArithmeticOrNone = getIRArithmetic(U, PointerTypes);
        if (not IRArithmeticOrNone.has_value())
          continue;

        IRArithmetic &IRPointerArithmetic = IRArithmeticOrNone.value();
        revng_log(ModelGEPLog, "IRPointerArithmetic " << IRPointerArithmetic);

        if (not IRPointerArithmetic.isAddress())
          continue;

        const auto &[BaseAddress, IRSum] = IRPointerArithmetic;
        const model::QualifiedType &PointeeType = BaseAddress.getPointeeType();

        if (PointeeType.isVoid()
            or PointeeType.is(model::TypeKind::RawFunctionType)
            or PointeeType.is(model::TypeKind::CABIFunctionType))
          continue;

        // Compute the type accessed by this use if any.
        std::optional<model::QualifiedType>
          AccessedTypeOnIR = getAccessedTypeOnIR(Cache,
                                                 U,
                                                 Model,
                                                 PointerTypes,
                                                 GEPifiedUsedTypes);

        // Now, compute the best GEP arguments for traversing the PointeeType.
        //
        // Notice that we have to try and access the pointer both via a
        // regular dereference and with the [] operator.
        // In order to do this, we have to create a fake array qualifier to
        // wrap around the PointeeType, and compute the best access in there.
        const model::Architecture::Values &Arch = Model.Architecture();
        model::QualifiedType FakeArray = PointeeType.getPointerTo(Arch);
        // Now replace the pointer qualifier with an array qualifier with
        // arbitrary maximum length. Don't use the max value for uint64_t
        // because it may overflow when computing the array size as array
        // length * element size.
        // TODO: MaxArrayLength is arbitrary. On the one hand we should make
        // it large enough to enable traversals with large indices without
        // falling back to out-of-bound raw-integer-arithmetic. On the other
        // hand if we make it too large we hit overflows on computations of
        // the total size of the array. So UINT32_MAX seemed like a good
        // compromise, but at some point we might need to handle this
        // properly.
        const uint64_t MaxArrayLength = std::numeric_limits<uint32_t>::max();
        auto LongArray = model::Qualifier::createArray(MaxArrayLength);
        FakeArray.Qualifiers().front() = std::move(LongArray);

        // Select among the computed TAPIndices the one which best fits the
        // IRPattern
        ModelGEPReplacementInfo GEPArgs = computeBest(FakeArray,
                                                      IRSum,
                                                      AccessedTypeOnIR,
                                                      VH);

        // Fix up the BaseType. This needs to contain the base type as per the
        // ModelGEP specification, not the fake array.
        GEPArgs.BaseType = PointeeType;
        revng_log(ModelGEPLog, "GEPArgs: " << GEPArgs);

        // If the result is doesn't have any Mismatched we can use it to
        // propagate information in GEPifiedUsedTypes and PointerTypes.
        if (GEPArgs.Mismatched.isZero()) {
          auto PointerToGEPArgs = GEPArgs.AccessedType.getPointerTo(Arch);
          GEPifiedUsedTypes.insert({ &U, PointerToGEPArgs });

          revng_log(ModelGEPLog, "Best GEPArgs: " << GEPArgs);

          // If IRSum is an address and I is an "address barrier"
          // instruction (e.g. an instruction such that pointer arithmetic does
          // not propagate through it), we need to check if we can still deduce
          // a rich pointer type for I starting from IRSum. An example of an
          // "address barrier" is a xor instruction (where we cannot deduce the
          // type of the xored value even if one of the operands has a known
          // pointer type); another example is a phi, where we can always deduce
          // that the phi has a rich pointer type if one of the incoming values
          // has a rich pointer type. The example of the xor is particularly
          // interesting, because one day we can think of starting to support it
          // for addresses that are built with masks, with small analyses. So
          // this is good customization point.
          //
          // In particular, we need to take care at least of the following
          // cases:
          // DONE:
          // - if I is a load and the loaded stuff is a pointer we have to set
          //   the type of the load
          // TODO:
          // - if I is a phi, we need to set the phi type
          //   - if one of the incoming has pointer type, we can take that. but
          //   what happens if many incoming have different pointer types, can
          //   we use a pointer to the parent type (the one that all should
          //   inherit from)?
          // - if I is a select instruction we can do something like the PHI
          // - if I is an alloca, I'm not sure what we can do
          if (auto *Load = dyn_cast<llvm::LoadInst>(&I)) {
            model::QualifiedType GEPType = getType(FakeArray,
                                                   GEPArgs.IndexVector,
                                                   VH);
            if (GEPType.isPointer())
              PointerTypes.insert({ Load, GEPType });
          }
        }

        Result.push_back({ &U, BaseAddress.getAddress(), std::move(GEPArgs) });
      }
    }
  }
  return Result;
}

class ModelGEPArgCache {

  std::map<model::QualifiedType, Constant *> GlobalModelGEPTypeArgs;

public:
  Value *getQualifiedTypeArg(const model::QualifiedType &QT, llvm::Module &M) {
    auto It = GlobalModelGEPTypeArgs.find(QT);
    if (It != GlobalModelGEPTypeArgs.end())
      return It->second;

    It = GlobalModelGEPTypeArgs.insert({ QT, serializeToLLVMString(QT, M) })
           .first;
    return It->second;
  }
};

static llvm::BasicBlock *getUniqueIncoming(Value *V, PHINode *Phi) {
  llvm::BasicBlock *Result = nullptr;

  for (auto [IncomingValue, IncomingBlock] :
       zip(Phi->incoming_values(), Phi->blocks())) {
    if (IncomingValue == V) {
      if (Result == nullptr) {
        // First matching incoming value, register incoming block
        Result = IncomingBlock;
      } else if (Result != IncomingBlock) {
        // We different incoming blocks for the same value
        return nullptr;
      }
    }
  }

  revng_assert(Result != nullptr, "V is not an incoming value of Phi");

  return Result;
}

struct MakeModelGEPPass : public FunctionPass {
public:
  static char ID;

  MakeModelGEPPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<FunctionMetadataCachePass>();
  }
};

bool MakeModelGEPPass::runOnFunction(llvm::Function &F) {
  bool Changed = false;

  // Skip non-isolated functions
  if (not FunctionTags::Isolated.isTagOf(&F))
    return Changed;

  revng_log(ModelGEPLog, "Make ModelGEP for " << F.getName());
  auto Indent = LoggerIndent(ModelGEPLog);

  auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
  auto &Cache = getAnalysis<FunctionMetadataCachePass>().get();

  model::VerifyHelper VH;
  auto GEPReplacements = makeGEPReplacements(F, *Model, VH, Cache);

  llvm::Module &M = *F.getParent();
  LLVMContext &Ctxt = M.getContext();
  IRBuilder<> Builder(Ctxt);
  ModelGEPArgCache TypeArgCache;

  // Create a function pool for AddressOf calls
  OpaqueFunctionsPool<TypePair> AddressOfPool(&M,
                                              /* PurgeOnDestruction */ false);
  if (not GEPReplacements.empty())
    initAddressOfPool(AddressOfPool, &M);

  llvm::IntegerType *PtrSizedInteger = getPointerSizedInteger(Ctxt, *Model);

  std::map<std::pair<Instruction *, Value *>, Value *> PhiIncomingsMaps;

  for (auto &[TheUseToGEPify, BaseAddress, GEPArgs] : GEPReplacements) {

    const auto &[BaseType, IndexVector, Mismatched, AccessedType] = GEPArgs;
    const auto &[MismatchedOffset, MismatchedIndices] = Mismatched;

    // Early exit for avoiding to emit a lot of &* in C.
    {
      if (IndexVector.empty())
        continue;

      const auto &[Index, AggregateTy] = IndexVector.front();
      revng_assert(AggregateTy == AggregateKind::Array);

      // This is not necessary or wrong strictly speaking, but if we don't bail
      // out on this case we end up generating a lot of &* in C.
      if (IndexVector.size() == 1 and Index.isZero())
        continue;
    }

    auto *UserInstr = cast<Instruction>(TheUseToGEPify->getUser());
    auto *V = TheUseToGEPify->get();

    bool IsPhi = isa<PHINode>(UserInstr);
    if (IsPhi) {
      auto It = PhiIncomingsMaps.find({ UserInstr, V });
      if (It != PhiIncomingsMaps.end()) {
        TheUseToGEPify->set(It->second);
        revng_log(ModelGEPLog,
                  "    `-> replaced with: " << dumpToString(It->second));
        Changed = true;
        continue;
      }
    }

    revng_log(ModelGEPLog, "GEPify use of: " << dumpToString(V));
    revng_log(ModelGEPLog, "  `-> use in: " << dumpToString(UserInstr));

    llvm::Type *UseType = TheUseToGEPify->get()->getType();
    llvm::Type *BaseAddrType = BaseAddress->getType();

    llvm::IntegerType *ModelGEPReturnedType;

    if (Mismatched.isZero()) {
      uint64_t PointeeSize = *AccessedType.size(VH);
      ModelGEPReturnedType = llvm::IntegerType::get(Ctxt, PointeeSize * 8);
    } else {
      // Calculate the size of the GEPPed field
      // If there is a remaining offset, we are returning something more
      // similar to a pointer than the actual value
      ModelGEPReturnedType = PtrSizedInteger;
    }

    auto *ModelGEPFunction = getModelGEP(M, ModelGEPReturnedType, BaseAddrType);

    // Build the arguments for the call to modelGEP
    SmallVector<Value *, 4> Args;
    Args.reserve(GEPArgs.IndexVector.size() + 2);

    // The first argument is always a pointer to a constant global variable
    // that holds the string representing the yaml serialization of the
    // qualified type of the base type of the modelGEP
    auto *BaseTypeConstantStrPtr = TypeArgCache.getQualifiedTypeArg(BaseType,
                                                                    M);
    Args.push_back(BaseTypeConstantStrPtr);

    // The second argument is the base address
    Args.push_back(BaseAddress);

    // Set the insert point for building the other arguments
    if (auto *PHIUser = dyn_cast<PHINode>(UserInstr)) {
      if (isa<llvm::Argument>(TheUseToGEPify->get())) {
        // Insert at the beginning of the function
        Builder.SetInsertPoint(F.getEntryBlock().getFirstNonPHI());
      } else if (llvm::BasicBlock *Incoming = getUniqueIncoming(V, PHIUser)) {
        // Insert at the end of the incoming block
        Builder.SetInsertPoint(Incoming->getTerminator());
      } else {
        // Insert at the end of the block of the user
        auto *IncomingInstruction = cast<Instruction>(TheUseToGEPify->get());
        auto *Terminator = IncomingInstruction->getParent()->getTerminator();
        Builder.SetInsertPoint(Terminator);
      }
    } else {
      Builder.SetInsertPoint(UserInstr);
    }

    // The other arguments are the indices in IndexVector
    for (auto &Group : llvm::enumerate(GEPArgs.IndexVector)) {
      const auto &[Index, AggregateTy] = Group.value();
      const auto &[ConstantIndex, InductionVariables] = Index;
      revng_assert(AggregateTy == AggregateKind::Array
                   or InductionVariables.empty());

      Value *IndexValue = nullptr;
      if (InductionVariables.empty() or not ConstantIndex.isNullValue()) {
        auto *Int64Type = llvm::IntegerType::get(Ctxt, 64 /*NumBits*/);
        IndexValue = ConstantInt::get(Int64Type, ConstantIndex);
      }

      for (const auto &[Coefficient, InductionVariable] : InductionVariables) {
        Value *Addend = InductionVariable;
        if (not Coefficient->isOne()) {
          auto *
            CoefficientConstant = ConstantInt::get(InductionVariable->getType(),
                                                   Coefficient->getZExtValue());
          Addend = Builder.CreateMul(CoefficientConstant, InductionVariable);
        }

        auto AddendBitWidth = Addend->getType()->getIntegerBitWidth();
        revng_assert(AddendBitWidth);

        if (Group.index() == 0 and AddendBitWidth != 64) {
          auto *Int64Type = llvm::IntegerType::getIntNTy(Addend->getContext(),
                                                         64);
          Addend = Builder.CreateZExtOrTrunc(Addend, Int64Type);
        }

        if (IndexValue) {

          if (auto BitWidth = IndexValue->getType()->getIntegerBitWidth();
              BitWidth > Addend->getType()->getIntegerBitWidth())
            Addend = Builder.CreateZExt(Addend, IndexValue->getType());

          IndexValue = Builder.CreateAdd(Addend, IndexValue);
        } else {
          IndexValue = Addend;
        }
      }

      Args.push_back(IndexValue);
    }

    Value *ModelGEPRef = Builder.CreateCall(ModelGEPFunction, Args);

    // If there is a remaining offset, we are returning something more
    // similar to a pointer than the actual value
    auto AddrOfReturnedType = Mismatched.isZero() ? UseType : PtrSizedInteger;

    // Inject a call to AddressOf
    auto *AddressOfFunctionType = getAddressOfType(AddrOfReturnedType,
                                                   ModelGEPReturnedType);
    auto *AddressOfFunction = AddressOfPool.get({ AddrOfReturnedType,
                                                  ModelGEPReturnedType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    auto *PointeeConstantStrPtr = TypeArgCache.getQualifiedTypeArg(AccessedType,
                                                                   M);
    Value *ModelGEPPtr = Builder.CreateCall(AddressOfFunction,
                                            { PointeeConstantStrPtr,
                                              ModelGEPRef });

    if (not Mismatched.isZero()) {
      // If the GEPArgs have a RestOff that is strictly positive, we have to
      // inject the remaining part of the pointer arithmetic as normal sums
      auto GEPResultBitWidth = ModelGEPPtr->getType()->getIntegerBitWidth();
      APInt OffsetToAdd = MismatchedOffset.zextOrTrunc(GEPResultBitWidth);
      if (not OffsetToAdd.isNullValue())
        ModelGEPPtr = Builder.CreateAdd(ModelGEPPtr,
                                        ConstantInt::get(Ctxt, OffsetToAdd));
      for (const auto &[Coefficient, Index] : MismatchedIndices) {
        ModelGEPPtr = Builder.CreateAdd(ModelGEPPtr,
                                        Builder.CreateMul(Index, Coefficient));
      }

      if (UseType->isPointerTy()) {
        // Convert the `AddressOf` result to a pointer in the IR if needed
        ModelGEPPtr = Builder.CreateIntToPtr(ModelGEPPtr, UseType);
      } else if (UseType != ModelGEPPtr->getType() and UseType->isIntegerTy()) {
        ModelGEPPtr = Builder.CreateZExt(ModelGEPPtr, UseType);
      }

      revng_assert(UseType == ModelGEPPtr->getType());
    }

    // Finally, replace the use to gepify with the call to the address of
    // modelGEP, plus the potential arithmetic we've just build.
    TheUseToGEPify->set(ModelGEPPtr);

    revng_log(ModelGEPLog,
              "    `-> replaced with: " << dumpToString(ModelGEPPtr));

    if (IsPhi)
      PhiIncomingsMaps[{ UserInstr, V }] = ModelGEPPtr;

    Changed = true;
  }

  revng::verify(F.getParent());

  return Changed;
}

char MakeModelGEPPass::ID = 0;

using Pass = MakeModelGEPPass;
static RegisterPass<Pass> X("make-model-gep",
                            "Pass that transforms address arithmetic into "
                            "calls to ModelGEP ",
                            false,
                            false);
