//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <limits>

#include "llvm/ADT/APInt.h"
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
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/VerifyHelper.h"
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
using llvm::Argument;
using llvm::BinaryOperator;
using llvm::CallInst;
using llvm::cast;
using llvm::Constant;
using llvm::ConstantExpr;
using llvm::ConstantInt;
using llvm::dyn_cast;
using llvm::FunctionPass;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::IRBuilder;
using llvm::isa;
using llvm::LLVMContext;
using llvm::LoadInst;
using llvm::None;
using llvm::PHINode;
using llvm::RegisterPass;
using llvm::ReturnInst;
using llvm::ReversePostOrderTraversal;
using llvm::SmallVector;
using llvm::StoreInst;
using llvm::Use;
using llvm::User;
using llvm::Value;
using model::CABIFunctionType;
using model::Qualifier;
using model::RawFunctionType;

using ModelTypesMap = std::map<const llvm::Value *, const model::QualifiedType>;

static Logger<> ModelGEPLog{ "make-model-gep" };

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

// We're trying to build a GEP summation in the form:
//     BaseAddress + sum( const_i * index_i)
// where BaseAddress is an llvm::Value, const_i are llvm::ConstantInt, and
// index_i are llvm::Values.
// This struct represent an element of the summation: const_i * index_i
struct ModelGEPSummationElement {
  ConstantInt *Coefficient = nullptr;
  Value *Index = nullptr;

  static bool isValid(const ModelGEPSummationElement &A) {
    return A.Coefficient != nullptr and A.Index != nullptr;
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelGEPSummationElement {\nCofficient:\n";
    if (Coefficient)
      OS << Coefficient->getValue().toString(10, true);
    else
      OS << "nullptr";
    OS << "\nIndex:\n";
    if (Index)
      Index->print(OS);
    else
      OS << "nullptr";
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using GEPSummationVector = SmallVector<ModelGEPSummationElement, 4>;

struct TypedBaseAddress {
  model::QualifiedType Type = {};
  Value *Address = nullptr;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "TypedBaseAddress{\nType:\n";
    serialize(OS, Type);
    OS << "\nAddress: ";
    if (Address)
      Address->print(OS);
    else
      OS << "nullptr";
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

// This struct represents an expression of the form:
//     BaseAddress + sum( const_i * index_i)
struct ModelGEPSummation {
  // If this has nullptr Address it means this summation does not represent an
  // address, but simply a summation of offsets.
  TypedBaseAddress BaseAddress = {};

  // If this is empty it means a zero offset.
  GEPSummationVector Summation = {};

  bool isAddress() const { return BaseAddress.Address != nullptr; }

  bool isValid() const {
    return llvm::all_of(Summation, ModelGEPSummationElement::isValid);
  }

  static ModelGEPSummation invalid() {
    return ModelGEPSummation{
      // The base address is unknown
      .BaseAddress = TypedBaseAddress{ .Type = {}, .Address = nullptr },
      // The summation has only one element, which is not valid, because it
      // does
      // not have a valid Index nor a valid Coefficient.
      .Summation = { ModelGEPSummationElement{ .Coefficient = nullptr,
                                               .Index = nullptr } }
    };
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelGEPSummation {\nBaseAddress: ";
    BaseAddress.dump(OS);
    OS << "\nSummation: {\n";
    for (const auto &SumElem : Summation) {
      SumElem.dump(OS);
      OS << '\n';
    }
    OS << "}\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using UseGEPSummationMap = std::map<Use *, ModelGEPSummation>;

struct IRAccessPattern {
  APInt BaseOffset = APInt(/*NumBits*/ 64, /*Value*/ 0);
  GEPSummationVector Indices = {};
  std::optional<model::QualifiedType> PointeeType = std::nullopt;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "IRAccessPattern {\nBaseOffset: " << BaseOffset.toString(10, true)
       << "\nIndices = {";
    for (const auto &I : Indices) {
      OS << "\n";
      I.dump(OS);
    }
    OS << "}\nPointeeType: ";
    if (PointeeType.has_value())
      serialize(OS, PointeeType.value());
    else
      OS << "std::nullopt";
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using UseTypeMap = std::map<llvm::Use *, model::QualifiedType>;

static IRAccessPattern
computeIRAccessPattern(FunctionMetadataCache &Cache,
                       const Use &U,
                       const ModelGEPSummation &GEPSum,
                       const model::Binary &Model,
                       const ModelTypesMap &PointerTypes,
                       const UseTypeMap &GEPifiedUseTypes) {
  using namespace model;
  revng_assert(GEPSum.isAddress());

  // First, prepare the BaseOffset and the Indices for the IRAccessPattern.
  GEPSummationVector IRPatternIndices;
  APInt BaseOff = APInt(/*NumBits*/ 64, /*Value*/ 0);

  // Accumulate all the constant SumElements into BaseOff, and all the others in
  // IRPatternIndices.
  for (const auto &SumElement : GEPSum.Summation) {
    const auto &[Coeff, Idx] = SumElement;
    revng_assert(Coeff->getValue().isNonNegative());

    // If this SumElement is a constant, update the BaseOff
    if (const auto &ConstIdx = dyn_cast<ConstantInt>(Idx)) {
      APInt Idx = ConstIdx->getValue();
      revng_assert(Coeff->getValue().isOneValue());
      revng_assert(Idx.isStrictlyPositive());
      revng_assert(Idx.getActiveBits() <= BaseOff.getBitWidth());
      BaseOff += Idx.zextOrTrunc(BaseOff.getBitWidth());
    } else {
      // Otherwise append the index to IRPatternIndices
      IRPatternIndices.push_back(SumElement);
    }
  }

  // Sort the IRPatternIndices so that strided accesses with larger strides come
  // first.
  const auto HasLargerStride = [](const ModelGEPSummationElement &LHS,
                                  const ModelGEPSummationElement &RHS) {
    return LHS.Coefficient->getValue().ugt(RHS.Coefficient->getValue());
  };
  llvm::sort(IRPatternIndices, HasLargerStride);

  // Now we're ready to initialize the IRAccessPattern
  IRAccessPattern IRPattern{ .BaseOffset = BaseOff.zextOrSelf(64),
                             .Indices = IRPatternIndices,
                             // Intially PointeeType is set to None, then we
                             // fill it if in some special cases where we have
                             // interesting information on the pointee
                             .PointeeType = std::nullopt };

  // The IRAccessPattern we've just initialized is not necessarily complete now.
  // We want to look at the user of U, to see if it gives us more information
  // about the PointeeType.

  if (auto *UserInstr = dyn_cast<Instruction>(U.getUser())) {
    if (auto *Load = dyn_cast<LoadInst>(UserInstr)) {
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
      IRPattern.PointeeType = QPointee;

    } else if (auto *Store = dyn_cast<StoreInst>(UserInstr)) {
      revng_log(ModelGEPLog, "User is Store");
      // If the user of U is a store, and U is the pointer operand, we know
      // that the pointee's size is equal to the size of the stored value.
      if (U.getOperandNo() == StoreInst::getPointerOperandIndex()) {
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
        IRPattern.PointeeType = QPointee;
      } else {
        revng_log(ModelGEPLog, "Use is pointer operand");
      }

    } else if (auto *Ret = dyn_cast<ReturnInst>(UserInstr)) {
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
          IRPattern.PointeeType = dropPointer(*SingleReturnType);
        }

      } else {
        auto *RetVal = Ret->getReturnValue();
        auto *StructTy = cast<llvm::StructType>(RetVal->getType());
        revng_log(ModelGEPLog, "Has many return types.");
        auto ReturnValuesCount = Layout.ReturnValues.size();
        revng_assert(StructTy->getNumElements() == ReturnValuesCount);

        // Assert that we're returning a proper struct, initialized with
        // struct initializers, but don't do anything here.
        const auto *Returned = cast<CallInst>(RetVal)->getCalledFunction();
        revng_assert(FunctionTags::StructInitializer.isTagOf(Returned));
      }

    } else if (auto *Call = dyn_cast<CallInst>(UserInstr)) {
      // If the user is a call, and it's calling an isolated function we want to
      // look at the argument types of the callee on the model, and use info
      // coming from them for initializing IRPattern.PointeeType

      revng_log(ModelGEPLog, "Call");
      const llvm::Function *CalledF = Call->getCalledFunction();

      if (CalledF and FunctionTags::StructInitializer.isTagOf(CalledF)) {

        // special case for struct initializers
        unsigned ArgNum = Call->getArgOperandNo(&U);

        const model::Function *CalledFType = llvmToModelFunction(Model,
                                                                 *CalledF);
        const model::Type *CalledPrototype = CalledFType->Prototype()
                                               .getConst();

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
            IRPattern.PointeeType = Pointee;
          }
        } else if (auto *CFT = dyn_cast<CABIFunctionType>(CalledPrototype)) {
          revng_log(ModelGEPLog, "Has CABIFunctionType prototype.");
          // TODO: we haven't handled return values of CABIFunctions yet
          revng_abort();
        } else {
          revng_abort("Function should have RawFunctionType or "
                      "CABIFunctionType");
        }

      } else if (FunctionTags::CallToLifted.isTagOf(Call)) {
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
              IRPattern.PointeeType = Pointee;
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
              IRPattern.PointeeType = Pointee;
            }
          } else {
            revng_log(ModelGEPLog, "IsCallee");
          }

        } else {
          revng_abort("Function should have RawFunctionType or "
                      "CABIFunctionType");
        }
      }

    } else if (auto *PHI = dyn_cast<PHINode>(UserInstr)) {
    }
  }

  return IRPattern;
}

struct ArrayInfo {
  APInt Stride = APInt(/*NumBits*/ 64, /*Value*/ 0);
  APInt NumElems = APInt(/*NumBits*/ 64, /*Value*/ 0);

  bool operator==(const ArrayInfo &) const = default;

  bool operator<(const ArrayInfo &Other) const {
    if (Stride.ult(Other.Stride))
      return true;
    if (Stride.ugt(Other.Stride))
      return false;
    if (NumElems.ult(Other.NumElems))
      return true;
    return false;
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ArrayInfo { .Stride = " << Stride.toString(10, true)
       << ", .NumElems = " << NumElems.toString(10, true) << "}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using ArrayInfoVector = SmallVector<ArrayInfo, 4>;

struct TypedAccessPattern {
  APInt BaseOffset = APInt(/*NumBits*/ 64, /*Value*/ 0);
  ArrayInfoVector Arrays = {};
  model::QualifiedType AccessedType = {};

  bool operator==(const TypedAccessPattern &) const = default;

  bool operator<(const TypedAccessPattern &Other) const {
    if (BaseOffset.ult(Other.BaseOffset))
      return true;
    if (BaseOffset.ugt(Other.BaseOffset))
      return false;

    if (Arrays < Other.Arrays)
      return true;
    if (Other.Arrays < Arrays)
      return false;

    return AccessedType < Other.AccessedType;
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "TypedAccessPattern {\nBaseOffset: " << BaseOffset.toString(10, true)
       << "\n";
    OS << "Arrays: {";
    for (const auto &AI : Arrays) {
      OS << "\n";
      AI.dump(OS);
    }
    OS << "\n}\nType: ";
    serialize(OS, AccessedType);
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

enum AggregateKind { Invalid, Struct, Union, Array };

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

struct ChildInfo {
  size_t ConstantIndex = 0ULL;
  llvm::Value *InductionVariable = nullptr;
  AggregateKind Type = AggregateKind::Invalid;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ChildInfo{\nConstantIndex: " << ConstantIndex
       << "\nType: " << toString(Type) << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using ChildIndexVector = SmallVector<ChildInfo, 4>;

class DifferenceScore {
  // Among two DifferenceScore, the one with PerfectTypeMatch set to true is
  // always the lowest.
  bool PerfectTypeMatch = false;

  // Boolean to mark out-of-range accesses
  bool InRange = false;

  // Higher Difference are for stuff that is farther apart from a perfect match
  // from the beginning of the type.
  size_t Difference = std::numeric_limits<size_t>::max();

  // Among two DifferenceScore, the one with ExactSize is considered closer.
  bool ExactSize = false;

  // This field represents how deep the type system was traversed to compute the
  // score. Scores with a higher depth are considered better (so lower
  // difference) because it means that the type system was traversed deeply.
  size_t Depth = std::numeric_limits<size_t>::min();

public:
  constexpr std::strong_ordering
  operator<=>(const DifferenceScore &Other) const {

    // If only one of the two is a perfect match, the difference is always
    // lower.
    if (PerfectTypeMatch != Other.PerfectTypeMatch)
      return PerfectTypeMatch ? std::strong_ordering::less :
                                std::strong_ordering::greater;

    // Here both are perfect matches or none is.

    // If one out of range, the difference is always higher
    if (InRange != Other.InRange)
      return InRange ? std::strong_ordering::less :
                       std::strong_ordering::greater;

    // Here both are in range or none is.

    // If the Difference is not the same, one of the two is closer to exact, so
    // we give precedence to that.
    if (auto Cmp = Difference <=> Other.Difference; Cmp != 0)
      return Cmp;

    // Here both have the same difference, e.g. they reach the same offset.

    // If the ExactSize is not the same, we give precedence to the one with
    // ExactSize, which means that it's closer in size to the
    // perfect match.
    if (ExactSize != Other.ExactSize)
      return ExactSize ? std::strong_ordering::less :
                         std::strong_ordering::greater;

    // Notice that in the following line the terms are inverted, because lower
    // depth needs to be scored "better" (so lower difference).
    return Other.Depth <=> Depth;
  }

  constexpr bool operator==(const DifferenceScore &Other) const = default;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "DifferenceScore { .PerfectTypeMatch = "
       << (PerfectTypeMatch ? "true" : "false")
       << ", .Difference = " << Difference
       << ", .ExactSize = " << (ExactSize ? "true" : "false")
       << ", .Depth = " << Depth
       << ", .InRange = " << (InRange ? "true" : "false") << "}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }

  static constexpr DifferenceScore max() { return DifferenceScore{}; }

  static constexpr DifferenceScore min() {
    DifferenceScore Result;
    Result.PerfectTypeMatch = true;
    Result.InRange = true;
    Result.Difference = std::numeric_limits<size_t>::min();
    Result.ExactSize = true;
    Result.Depth = std::numeric_limits<size_t>::max();
    return Result;
  }

  static constexpr DifferenceScore
  nestedOutOfBound(size_t DiffScore, size_t Depth) {
    DifferenceScore Result;
    Result.PerfectTypeMatch = false;
    Result.InRange = false;
    Result.Difference = DiffScore;
    Result.ExactSize = false;
    Result.Depth = Depth;
    return Result;
  }

  static constexpr DifferenceScore outOfBound(size_t DiffScore) {
    return nestedOutOfBound(DiffScore, std::numeric_limits<size_t>::min());
  }

  static constexpr DifferenceScore
  inRange(bool PerfectMatch, uint64_t DiffScore, bool SameSize, size_t Depth) {
    DifferenceScore Result;
    Result.PerfectTypeMatch = PerfectMatch;
    Result.InRange = true;
    Result.Difference = DiffScore;
    Result.ExactSize = SameSize;
    Result.Depth = Depth;
    return Result;
  }
};

static constexpr auto nestedOutOfBound = &DifferenceScore::nestedOutOfBound;
static constexpr auto outOfBound = &DifferenceScore::outOfBound;

// Some static asserts about ordering of DifferenceScore
//
// Basically we have:
// 1) outOfBound(x) == nestedOutOfBound(x, 0)
// 2) min < nestedOutOfBound(x, y) <= max
// 3) at the previous point, the <= with max is == when:
// 3.1) y == 0 and
// 3.2) x == std::numeric_limits<size_t>::max()

static_assert(DifferenceScore::max() == nestedOutOfBound(SIZE_MAX, 0));
static_assert(DifferenceScore::max() == outOfBound(SIZE_MAX));

static_assert(DifferenceScore::min() < DifferenceScore::max());

static_assert(outOfBound(0) == nestedOutOfBound(0, 0));
static_assert(outOfBound(1) == nestedOutOfBound(1, 0));
static_assert(outOfBound(1) > nestedOutOfBound(1, 2));
static_assert(outOfBound(5) > nestedOutOfBound(1, 2));
static_assert(outOfBound(0) < nestedOutOfBound(1, 2));

static_assert(DifferenceScore::min() < nestedOutOfBound(1, 1));
static_assert(DifferenceScore::max() > nestedOutOfBound(1, 1));

static_assert(DifferenceScore::min() < outOfBound(0));
static_assert(DifferenceScore::max() > outOfBound(0));

struct ScoredIndices {
  // The score is optional, nullopt means that the difference score is infinity
  DifferenceScore Score = DifferenceScore::max();
  ChildIndexVector Indices = {};

  static ScoredIndices outOfBound(size_t DiffScore) {
    return ScoredIndices{ .Score = DifferenceScore::outOfBound(DiffScore),
                          .Indices{} };
  }

  static ScoredIndices
  nestedOutOfBound(size_t DiffScore, size_t Depth, ChildIndexVector &&Indices) {
    revng_assert(Depth <= Indices.size());
    if (Depth < Indices.size())
      Indices.resize(Depth);
    return ScoredIndices{ .Score = DifferenceScore::nestedOutOfBound(DiffScore,
                                                                     Depth),
                          .Indices = std::move(Indices) };
  }
};

using TAPIndicesPair = std::pair<TypedAccessPattern, ChildIndexVector>;

static ScoredIndices differenceScore(const model::QualifiedType &BaseType,
                                     const TAPIndicesPair &TAPWithIndices,
                                     const IRAccessPattern &IRAP,
                                     model::VerifyHelper &VH) {

  const auto &[TAP, ChildIndices] = TAPWithIndices;
  ChildIndexVector ResultIndices = ChildIndices;

  APInt RestOff = IRAP.BaseOffset;

  size_t BaseSize = *BaseType.size(VH);
  if (IRAP.BaseOffset.uge(BaseSize))
    return ScoredIndices::outOfBound(RestOff.getZExtValue());

  revng_assert(TAP.BaseOffset.ult(BaseSize));
  revng_assert((TAP.BaseOffset + *TAP.AccessedType.size(VH)).ule(BaseSize));

  auto ArrayInfoIt = TAP.Arrays.begin();
  auto ArrayInfoEnd = TAP.Arrays.end();

  auto IRAPIndicesIt = IRAP.Indices.begin();
  auto IRAPIndicesEnd = IRAP.Indices.end();

  model::QualifiedType NestedType = BaseType;

  size_t Depth = 0;
  for (auto &ChildID : ResultIndices) {
    model::QualifiedType Normalized = peelConstAndTypedefs(NestedType);

    // Should not be a pointer, because pointers don't have children on the
    // type system, which means that we shouldn't have a ChildId at this
    // point.
    revng_assert(not Normalized.isPointer());

    switch (ChildID.Type) {

    case Struct: {
      revng_assert(not Normalized.isArray());
      auto *Const = Normalized.UnqualifiedType().getConst();
      auto *S = cast<model::StructType>(Const);
      revng_assert(not ChildID.InductionVariable);
      size_t FieldOffset = ChildID.ConstantIndex;

      // If the RestOff is less than the field offset, it means that the IRAP
      // does not have enough offset to reach the field of the struct that is
      // required from the TAPWithIndices.
      // This should never happen.
      revng_assert(RestOff.uge(FieldOffset));
      RestOff -= FieldOffset;
      NestedType = S->Fields().at(FieldOffset).Type();
    } break;

    case Union: {
      revng_assert(not Normalized.isArray());
      auto *Const = Normalized.UnqualifiedType().getConst();
      auto *U = cast<model::UnionType>(Const);
      revng_assert(not ChildID.InductionVariable);
      size_t FieldID = ChildID.ConstantIndex;
      NestedType = U->Fields().at(FieldID).Type();
    } break;

    case Array: {
      revng_assert(Normalized.isArray());
      revng_assert(ArrayInfoIt != ArrayInfoEnd);

      const auto ArrayQualEnd = Normalized.Qualifiers().end();
      const auto ArrayQualIt = llvm::find_if(Normalized.Qualifiers(),
                                             model::Qualifier::isArray);
      revng_assert(ArrayQualIt != ArrayQualEnd);

      revng_assert(ArrayQualIt->Size() == ArrayInfoIt->NumElems);

      // Compute the index of the accessed element of the array.
      auto ArrayStride = ArrayInfoIt->Stride;
      auto MaxBitWidth = std::max(RestOff.getBitWidth(),
                                  ArrayStride.getBitWidth());
      APInt ElemIndex = APInt(MaxBitWidth, 0);
      APInt OffInElem = APInt(MaxBitWidth, 0);
      APInt::udivrem(RestOff.zextOrTrunc(MaxBitWidth),
                     ArrayStride.zextOrTrunc(MaxBitWidth),
                     ElemIndex,
                     OffInElem);

      if (ElemIndex.uge(ArrayInfoIt->NumElems)) {
        // If IRAP is trying to access an element that is larger than the
        // array size, we have to bail out, marking this as out of bound.
        return ScoredIndices::nestedOutOfBound(RestOff.getZExtValue(),
                                               Depth,
                                               std::move(ResultIndices));
      }

      if (IRAPIndicesIt == IRAPIndicesEnd) {
        // If the IRAP does not have strided accesses anymore, we have to
        // perform the access at a constant offset.
        // Assert that there's no Induction Variable.
        revng_assert(not ChildID.InductionVariable);

      } else {
        // Here we still hav some IRAPIndices, so we're modeling a strided
        // access and we must have a variable Index.
        const auto &[IRCoefficient, IRIndex] = *IRAPIndicesIt;
        revng_assert(IRIndex, "IRAPIndex not present");
        revng_assert(not llvm::isa<Constant>(IRIndex),
                     "IRAPIndex does not represent a strided access");

        // The IRAccessPattern should always have a Coefficient that is lower
        // than or equal than the array stride, otherwise we'de be modeling a
        // traversal that breakes type safety, which is what we're trying to
        // avoid.
        auto &IRCoefficientVal = IRCoefficient->getValue();
        auto IRCoefficientResized = IRCoefficientVal.zextOrTrunc(MaxBitWidth);
        auto ArrayStrideResized = ArrayStride.zextOrTrunc(MaxBitWidth);
        revng_assert(IRCoefficientResized.ule(ArrayStrideResized));

        if (IRCoefficientResized == ArrayStrideResized) {
          // If we're traversing an array with the same stride on the IR and on
          // the model, then the induction variable must be the same.
          revng_assert(IRIndex == ChildID.InductionVariable);
        } else {
          // If we're traversing an array with a smaller stride on the IR,
          // than the model, then the InductionVariable on the IR should be
          // ignored, and we should be accessing the model array at a fixed
          // index, without an induction variable on the model.
          revng_assert(not ChildID.InductionVariable);
        }

        ++IRAPIndicesIt;
      }

      RestOff = OffInElem;

      NestedType = model::QualifiedType(Normalized.UnqualifiedType(),
                                        { std::next(ArrayQualIt),
                                          ArrayQualEnd });

      ++ArrayInfoIt;
    } break;

    default:
      revng_abort();
    }

    // Increase the depth for each element in the type system that was traversed
    // succesfully.
    ++Depth;
  }

  bool PerfectMatch = IRAP.PointeeType.has_value()
                      and IRAP.PointeeType.value() == TAP.AccessedType;

  bool SameSize = false;
  if (IRAP.PointeeType.has_value()) {
    std::optional<uint64_t> TAPSize = TAP.AccessedType.trySize(VH);
    std::optional<uint64_t> IRAPSize = IRAP.PointeeType.value().trySize(VH);
    SameSize = TAPSize.value_or(0ull) == IRAPSize.value_or(0ull);
  }

  return ScoredIndices{
    .Score = DifferenceScore::inRange(PerfectMatch,
                                      RestOff.getZExtValue(),
                                      SameSize,
                                      Depth),
    .Indices = std::move(ResultIndices),
  };
}

struct ModelGEPArgs {
  TypedBaseAddress BaseAddress = {};
  ChildIndexVector IndexVector = {};
  APInt RestOff;
  model::QualifiedType PointeeType = {};

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelGEPArgs {\nBaseAddress:\n";
    BaseAddress.dump(OS);
    OS << "\nIndexVector: {";
    for (const auto &C : IndexVector) {
      OS << "\n";
      C.dump(OS);
    }
    OS << "}\nRestOff: " << RestOff << "\nPointeeType: ";
    serialize(OS, PointeeType);
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

static std::optional<model::QualifiedType>
getType(const model::QualifiedType &BaseType,
        const ChildIndexVector &IndexVector,
        APInt RestOff,
        model::VerifyHelper &VH) {
  std::optional<model::QualifiedType> CurrType = std::nullopt;

  if (RestOff.ugt(0))
    return CurrType;

  CurrType = BaseType;
  for (const auto &[FixedIndex, InductionVariable, AggregateType] :
       IndexVector) {

    switch (AggregateType) {

    case AggregateKind::Struct: {

      CurrType = peelConstAndTypedefs(CurrType.value());
      auto *S = cast<model::StructType>(CurrType->UnqualifiedType().getConst());
      revng_assert(not InductionVariable);
      CurrType = S->Fields().at(FixedIndex).Type();

    } break;

    case AggregateKind::Union: {

      CurrType = peelConstAndTypedefs(CurrType.value());
      auto *U = cast<model::UnionType>(CurrType->UnqualifiedType().getConst());
      revng_assert(not InductionVariable);
      CurrType = U->Fields().at(FixedIndex).Type();

    } break;

    case AggregateKind::Array: {

      auto ArrayQualIt = CurrType->Qualifiers().begin();

      do {
        CurrType = peelConstAndTypedefs(CurrType.value());

        ArrayQualIt = llvm::find_if(CurrType->Qualifiers(),
                                    model::Qualifier::isArray);

        // Assert that we're not skipping any pointer qualifier.
        // That would mean that the GEPArgs.IndexVector is broken w.r.t. the
        // GEPArgs.BaseAddress.
        revng_assert(not std::any_of(CurrType->Qualifiers().begin(),
                                     ArrayQualIt,
                                     model::Qualifier::isPointer));

      } while (ArrayQualIt == CurrType->Qualifiers().end());

      revng_assert(ArrayQualIt->Size() > FixedIndex);
      // For arrays we don't need to look at the value of the index, we just
      // unwrap the array and go on.
      CurrType = model::QualifiedType(CurrType->UnqualifiedType(),
                                      { std::next(ArrayQualIt),
                                        CurrType->Qualifiers().end() });

    } break;

    default:
      revng_abort();
    }
  }

  return CurrType;
}

static DifferenceScore lowerBound(const model::QualifiedType &BaseType,
                                  const IRAccessPattern &IRPattern,
                                  model::VerifyHelper &VH) {

  auto BaseSizeOrNone = static_cast<std::optional<size_t>>(BaseType.size(VH));
  revng_assert(BaseSizeOrNone.has_value());
  size_t BaseSize = BaseSizeOrNone.value();
  revng_assert(BaseSize);

  bool IRHasPointee = IRPattern.PointeeType.has_value();

  auto PointeeSize = IRHasPointee ? *IRPattern.PointeeType->trySize(VH) : 0ULL;

  if ((IRPattern.BaseOffset + PointeeSize).uge(BaseSize))
    return DifferenceScore::outOfBound(IRPattern.BaseOffset.getZExtValue());

  // Assume we can find a perfect match with the same size unless we find strong
  // evidence of the opposite.
  bool PerfectMatch = true;
  bool SameSize = true;

  // If the IRPattern has not pointee information, we'll never reach a situation
  // where SameSize nor PerfectMatch is true
  if (not IRHasPointee) {
    PerfectMatch = false;
    SameSize = false;
  }

  return DifferenceScore::inRange(PerfectMatch,
                                  /*DiffScore*/ 0,
                                  SameSize,
                                  /*Depth*/ std::numeric_limits<size_t>::max());
}

static RecursiveCoroutine<TAPIndicesPair>
computeBestTAP(model::QualifiedType BaseType,
               const IRAccessPattern &IRPattern,
               llvm::LLVMContext &Ctxt,
               model::VerifyHelper &VH) {
  revng_log(ModelGEPLog, "Computing Best TAP for IRAP: " << IRPattern);
  revng_assert(not BaseType.is(model::TypeKind::RawFunctionType)
               and not BaseType.is(model::TypeKind::CABIFunctionType));

  TypedAccessPattern BaseTAP = {
    // The BaseOffset is 0, since this TAP represents an access to the
    // entire BaseType starting from BaseType itself.
    .BaseOffset = APInt(/*NumBits*/ 64, /*Value*/ 0),
    // We have no arrays info, since this TAP represents an access to the
    // entire BaseType starting from BaseType itself.
    .Arrays = {},
    // The pointee is just the BaseType
    .AccessedType = BaseType,
  };

  // Baseline TAP with empty ChildIndexVector, representing the access to the
  // BaseType.
  auto Result = std::make_pair(BaseTAP, ChildIndexVector{});

  // Running variable to hold the best score. It will guide the branch-and-bound
  // algorithm for computing the best TypedAccessPattern with indices.
  // Compute the BaseScore e.g. the score of the traversal that ends at
  // BaseType (which is basically a non-traversal). This is the baseline to
  // compare with the scores of the children.
  auto BestScore = DifferenceScore::max();
  const auto &[BaseScore,
               BaseIndices] = differenceScore(BaseType, Result, IRPattern, VH);
  BestScore = std::move(BaseScore);
  Result.second = std::move(BaseIndices);

  // If we've reached a primitive type or an enum type we're done. The
  // BaseScore computed above is enough and we don't need to traverse
  // anything.
  // The same holds if we've reached a pointer, because the pointee does not
  // reside into the BaseType, it's only referenced by it.
  if (BaseType.isPrimitive() or BaseType.isPointer()
      or BaseType.is(model::TypeKind::EnumType))
    rc_return Result;

  // Here BaseType is either an array, a struct, a union, or a typedef.
  // Unwrap const and typedefs, because ModelGEPs just see through them.
  // Unwrapping const and typedefs can potentially lose information, but it's
  // fine to do it here, because after traversing stuff we only ever use either
  // the root type or the leaf type, and both are the same.
  BaseType = peelConstAndTypedefs(BaseType);

  // In all the other cases (arrays and const) we need to unwrap the first layer
  // (qualifier) and keep looking for other TAPs that might be generated.
  auto QBeg = BaseType.Qualifiers().begin();
  auto QEnd = BaseType.Qualifiers().end();
  if (QBeg != QEnd) {
    revng_log(ModelGEPLog, "Array, look at elements");
    auto ArrayIndent = LoggerIndent{ ModelGEPLog };

    auto &ArrayQualifier = *QBeg;
    revng_assert(model::Qualifier::isArray(ArrayQualifier));

    auto ElementType = model::QualifiedType(BaseType.UnqualifiedType(),
                                            { std::next(QBeg), QEnd });

    uint64_t NElems = ArrayQualifier.Size();
    std::optional<uint64_t> MaybeElementTypeSize = ElementType.size(VH);
    revng_assert(MaybeElementTypeSize.has_value()
                 and MaybeElementTypeSize.value());
    uint64_t ElementSize = MaybeElementTypeSize.value();

    uint64_t ArraySize = ElementSize * NElems;
    if (IRPattern.BaseOffset.uge(ArraySize))
      rc_return Result;

    // Set up the IRPattern we need for computing scores in the inner array
    // element. We initialize it to the IRPattern but we'll have to adjust it to
    // peel away the array access.
    revng_assert(IRPattern.BaseOffset.getBitWidth() == 64);
    IRAccessPattern ElemIRPattern = IRPattern;

    APInt APElementSize = APInt(/*NumBits*/ 64, /*value*/ ElementSize);

    // Set up the auxiliary variables that will later be used for updating the
    // result if the array traversal turns out to be the new best TAP.
    // We compute them here, while also updating ElemIRPattern, because the two
    // computations are related and we want to avoid redoing the work later on.
    APInt FixedElementIndex = APInt(/*NumBits*/ 64, /*value*/ ElementSize);
    llvm::Value *InductionVariable = nullptr;

    if (not ElemIRPattern.Indices.empty()) {
      // Unwrap the top layer of array indices, since we're traversing the
      // array, but only do it if the current layer of array accesses we're
      // peeling has a Coefficient that matches the element size of the array.
      // If we unwrap an array traversal with non-constant index, that's
      // the induction variable, and we have to track it.
      const auto &[Coefficient, Index] = ElemIRPattern.Indices.front();

      // Coefficient represents the stride of the pointer arithemtic pattern on
      // the IR. If it's larger than ElementSize on the model it means that
      // we're not accessing the array with an induction variable that
      // increments one by one.
      // This can happen in two scenarios:
      // - wild access, that we want to discard
      // - every n element, instead of every element, that we want to handle.
      if (Coefficient->getValue().ugt(ElementSize)) {
        revng_assert(Index);

        APInt NumMultipleElements = APInt(/*NumBits*/ 64, /*value*/ 0);
        APInt Remainder = APInt(/*NumBits*/ 64, /*value*/ 0);

        APInt::udivrem(Coefficient->getValue().zextOrSelf(64),
                       APElementSize,
                       NumMultipleElements,
                       Remainder);
        if (Remainder.getBoolValue()) {
          // This is not accessing an exact size of elements at each stride.
          // Just skip this.
          rc_return Result;
        } else {
          // TDOO: in principle we'd like to be able to save the
          // NumMultipleElement in the Result, so that we can emit array
          // accesses in the form: array[constant + induction_var * multiplier].
          rc_return Result;
        }

      } else {
        revng_assert(not isa<ConstantInt>(Index));

        if (Coefficient->getValue() == ElementSize) {
          InductionVariable = Index;
          ElemIRPattern.Indices.erase(ElemIRPattern.Indices.begin());
        }
      }
    }

    // Update ElemIRPattern.BaseOffset while also computing FixedElemIndex if
    // necessary.
    APInt::udivrem(IRPattern.BaseOffset,
                   APElementSize,
                   FixedElementIndex,
                   ElemIRPattern.BaseOffset);

    DifferenceScore LowerBound = lowerBound(ElementType, ElemIRPattern, VH);
    revng_log(ModelGEPLog, "lowerBound: " << LowerBound);

    if (bool CanImprove = LowerBound <= BestScore; CanImprove) {

      // We have estimated that accessing the element of the array might
      // yield a better score.
      // Let's compute the best TAP with indices, for accessing an element.
      TAPIndicesPair ElementResult = rc_recur computeBestTAP(ElementType,
                                                             ElemIRPattern,
                                                             Ctxt,
                                                             VH);

      // Now let's see if the best TAP of on the array element is actually
      // better than the current best.
      const auto &[ElementScore,
                   ElementIndices] = differenceScore(ElementType,
                                                     ElementResult,
                                                     ElemIRPattern,
                                                     VH);
      revng_assert(ElementScore >= LowerBound);
      // If it's better or equal, udpate the running best.
      // The "or equal" is important, because we score better the traversals
      // that go deeper in the type system.
      if (auto ElementCmp = ElementScore <=> BestScore; ElementCmp <= 0) {
        if (ElementCmp < 0) {
          revng_log(ModelGEPLog, "New BestScore: " << ElementScore);
          BestScore = ElementScore;
        }

        // Set Result to ElementResult. Then we'll have to update it with the
        // information about the array we've just traversed.
        Result = ElementResult;

        // Build the ArrayInfo associated to the array we're handling, and
        // prepend it to the Arrays of the Result, to represent the traversal
        // of this array.
        Result.first.Arrays.insert(Result.first.Arrays.begin(),
                                   ArrayInfo{
                                     .Stride = APInt(/*NumBits*/ 64,
                                                     /*Value*/ ElementSize),
                                     .NumElems = APInt(/*NumBits*/ 64,
                                                       /*Value*/ NElems) });

        // Build the child info associated to the array we're
        // handling, and prepend it to the child ids in the Result.
        Result.second
          .insert(Result.second.begin(),
                  ChildInfo{ .ConstantIndex = FixedElementIndex.getZExtValue(),
                             .InductionVariable = InductionVariable,
                             .Type = AggregateKind::Array });
      }
    }

    rc_return Result;
  }

  // We have no qualifiers here, just match struct and unions
  const model::Type *BaseT = BaseType.UnqualifiedType().getConst();
  switch (BaseT->Kind()) {

  case model::TypeKind::StructType: {

    revng_log(ModelGEPLog, "Struct, look at fields");
    auto StructIndent = LoggerIndent{ ModelGEPLog };

    const auto *S = cast<model::StructType>(BaseT);

    auto FieldBegin = S->Fields().begin();
    // Let's detect the leftmost field that starts later than the maximum offset
    // reachable by the IRPattern. This is the first element that we don't want
    // to compare, because it's not a valid traversal for the given IR pattern.
    auto FieldIt = S->Fields().upper_bound(IRPattern.BaseOffset.getZExtValue());

    enum {
      InitiallyImproving,
      StartedDegrading,
    } Status = InitiallyImproving;

    for (const model::StructField &Field :
         llvm::reverse(llvm::make_range(FieldBegin, FieldIt))) {

      revng_log(ModelGEPLog, "Field at offset: " << Field.Offset());
      auto FieldIndent = LoggerIndent{ ModelGEPLog };

      auto &FieldType = Field.Type();
      const auto FieldOffset = Field.Offset();
      revng_assert(IRPattern.BaseOffset.uge(FieldOffset));

      IRAccessPattern FieldAccessPattern = IRPattern;
      FieldAccessPattern.BaseOffset -= FieldOffset;

      DifferenceScore LowerBound = lowerBound(FieldType,
                                              FieldAccessPattern,
                                              VH);
      revng_log(ModelGEPLog, "lowerBound: " << LowerBound);
      bool CanImprove = LowerBound <= BestScore;

      // TODO: if this assertion never trigger, we could inject an early exit
      // from this loop on struct fields, so that when we start degrading we
      // just exit.
      // Even better would be to actually verify that this cannot happen by
      // design, and just remove the assertion and add the early exit.
      revng_assert(not CanImprove or Status == InitiallyImproving);
      if (not CanImprove)
        Status = StartedDegrading;

      if (CanImprove) {
        // We have estimated that accessing the struct field may yield a better
        // score. Let's compute the best TAP with indices, for accessing the
        // field.
        TAPIndicesPair FieldResult = rc_recur computeBestTAP(FieldType,
                                                             FieldAccessPattern,
                                                             Ctxt,
                                                             VH);

        // Now let's see if the best TAP of the field is actually better than
        // the current best.
        const auto &[FieldScore,
                     FieldIndices] = differenceScore(FieldType,
                                                     FieldResult,
                                                     FieldAccessPattern,
                                                     VH);
        revng_assert(FieldScore >= LowerBound);

        // If it's better or equal, udpate the running best.
        // The "or equal" is important, because we score better the traversals
        // that go deeper in the type system.
        if (auto FieldCmp = FieldScore <=> BestScore; FieldCmp <= 0) {
          if (FieldCmp < 0) {
            revng_log(ModelGEPLog, "New BestScore: " << FieldScore);
            BestScore = FieldScore;
          }

          // Set Result to FieldResult. Then we'll have to update it with the
          // information about the struct we've just traversed.
          Result = FieldResult;

          // Fixup the base offset
          Result.first.BaseOffset += FieldOffset;

          // Build the child info associated to the array we're
          // handling, and prepend it to the child ids in the Result.
          Result.second.insert(Result.second.begin(),
                               ChildInfo{ .ConstantIndex = Field.Offset(),
                                          .Type = AggregateKind::Struct });
        }
      }
    }
  } break;

  case model::TypeKind::UnionType: {

    revng_log(ModelGEPLog, "Union, look at fields");
    auto UnionIndent = LoggerIndent{ ModelGEPLog };

    const auto *U = cast<model::UnionType>(BaseT);

    for (const model::UnionField &Field : U->Fields()) {

      revng_log(ModelGEPLog, "Field ID: " << Field.Index());
      auto FieldIndent = LoggerIndent{ ModelGEPLog };

      auto &FieldType = Field.Type();

      DifferenceScore LowerBound = lowerBound(FieldType, IRPattern, VH);
      revng_log(ModelGEPLog, "lowerBound: " << LowerBound);

      if (bool CanImprove = LowerBound <= BestScore; CanImprove) {

        // We have estimated that accessing the union field may yield a better
        // score. Let's compute the best TAP with indices, for accessing the
        // field.
        TAPIndicesPair FieldResult = rc_recur computeBestTAP(FieldType,
                                                             IRPattern,
                                                             Ctxt,
                                                             VH);

        // Now let's see if the best TAP of the field is actually better than
        // the current best.
        const auto &[FieldScore, FieldIndices] = differenceScore(FieldType,
                                                                 FieldResult,
                                                                 IRPattern,
                                                                 VH);
        revng_assert(FieldScore >= LowerBound);

        // If it's better or equal, udpate the running best.
        // The "or equal" is important, because we score better the traversals
        // that go deeper in the type system.
        if (auto FieldCmp = FieldScore <=> BestScore; FieldCmp <= 0) {
          if (FieldCmp < 0) {
            revng_log(ModelGEPLog, "New BestScore: " << FieldScore);
            BestScore = FieldScore;
          }

          // Set Result to FieldResult. Then we'll have to update it with the
          // information about the union we've just traversed.
          Result = FieldResult;

          // Build the child info associated to the array we're
          // handling, and prepend it to the child ids in the Result.
          Result.second.insert(Result.second.begin(),
                               ChildInfo{ .ConstantIndex = Field.Index(),
                                          .Type = AggregateKind::Union });
        }
      }
    }
  } break;

  default:
    revng_abort();
  }

  rc_return Result;
}

static ModelGEPArgs makeBestGEPArgs(const TypedBaseAddress &TBA,
                                    const IRAccessPattern &IRPattern,
                                    const model::Binary &Model,
                                    model::VerifyHelper &VH) {
  LLVMContext &Ctxt = TBA.Address->getContext();

  revng_log(ModelGEPLog, "===============================");
  revng_log(ModelGEPLog, "makeBestGEPArgs for TBA: " << TBA);
  auto MakeBestGEPArgsIndent = LoggerIndent(ModelGEPLog);

  const TAPIndicesPair BestTAPsWithIndices = computeBestTAP(TBA.Type,
                                                            IRPattern,
                                                            Ctxt,
                                                            VH);
  const auto &[BestTAP, BestIndices] = BestTAPsWithIndices;

  // Setup a vector of indices to fill up. Most of them will be copies
  // straight from BestIndices, except for those representing array accesses,
  // that will need to be filled up with the actual Value, representing the
  // index of the array access.
  ChildIndexVector Indices = {};
  Indices.reserve(BestIndices.size());

  // A variable to hold the current type we have reached while traversing the
  // type system starting from TBA, while looking for the proper indices to
  // represent the array accesses.
  model::QualifiedType CurrentType = TBA.Type;

  // Holds the remaning constant offset we need to traverse to complete the
  // traversal of IRAccessPattern
  APInt RestOff = IRPattern.BaseOffset;

  auto TAPArrayIt = BestTAP.Arrays.begin();
  auto TAPArrayEnd = BestTAP.Arrays.end();

  revng_log(ModelGEPLog, "Initial RestOff: " << RestOff.toString(10, true));
  revng_log(ModelGEPLog, "Num indices: " << BestIndices.size());
  for (const auto &Id : BestIndices) {
    revng_log(ModelGEPLog, "RestOff: " << RestOff.toString(10, true));
    revng_log(ModelGEPLog, "Id: " << Id);

    Indices.push_back(Id);
    auto &Back = Indices.back();

    switch (Id.Type) {

    case AggregateKind::Array: {
      revng_assert(Back.Type == AggregateKind::Array);
      revng_assert(CurrentType.isArray());

      model::QualifiedType Array = peelConstAndTypedefs(CurrentType);
      auto ArrayQualIt = Array.Qualifiers().begin();
      auto QEnd = Array.Qualifiers().end();

      revng_assert(ArrayQualIt != QEnd);
      revng_assert(model::Qualifier::isArray(*ArrayQualIt));

      auto *Unqualified = Array.UnqualifiedType().get();

      model::QualifiedType
        ElementType = model::QualifiedType(Model.getTypePath(Unqualified),
                                           { std::next(ArrayQualIt), QEnd });

      // We've found the first array qualifier, for which we don't know the
      // index that is being accessed. That information is stored in the
      // IRAccessPattern indices.
      // We have to unwrap it and put data about it into Indices.

      // First of all, the rest of the offset needs to be smaller than the
      // array type size.
      revng_assert(RestOff.ule(*CurrentType.size(VH)));

      // Second, the TAP needs to still have non-consumed info associated to
      // arrays
      revng_assert(TAPArrayIt != TAPArrayEnd);

      // The array in BestTAP that we're unwrapping has a stride equal to
      // the size of this array element.
      uint64_t ElementSize = *ElementType.size(VH);
      revng_assert(TAPArrayIt->Stride == ElementSize);

      // If the remaining offset is larger than or equal to an element size,
      // we have to compute the exact index of the element that is being
      // accessed
      auto MaxBitWidth = std::max(RestOff.getBitWidth(), 64U);
      APInt ElementIndex = APInt(MaxBitWidth, 0);
      APInt::udivrem(RestOff.zextOrTrunc(MaxBitWidth),
                     APInt(/*bitwidth*/ MaxBitWidth, /*value*/ ElementSize),
                     ElementIndex,
                     RestOff);
      revng_assert(Back.ConstantIndex == ElementIndex.getZExtValue());

      // After we're done with an array, we update CurrentType and continue to
      // the next iteration of the for loop on BestIndices, because we could
      // have another array index and another array qualifier left in
      // CurrentType
      CurrentType = ElementType;
      revng_assert(RestOff.ule(*CurrentType.size(VH)));

      // We also omve the TAPArrayIt to point to the next array info available
      // in BestTAP
      ++TAPArrayIt;
      continue;

    } break;

    case AggregateKind::Struct: {
      revng_assert(CurrentType.is(model::TypeKind::StructType));
      auto QualifiedStruct = peelConstAndTypedefs(CurrentType);
      const auto *Unqualified = QualifiedStruct.UnqualifiedType().getConst();
      const auto *Struct = llvm::cast<model::StructType>(Unqualified);

      // Index represents the offset of a field in the struct
      revng_assert(not Back.InductionVariable);
      uint64_t FieldOff = Back.ConstantIndex;

      // The offset of the field should be smaller or equal to the remaining
      // offset. If it's not it means that the IRAP has not sufficient offset
      // to reach the pattern described by TAP. This should never happen.
      revng_assert(RestOff.uge(FieldOff));

      APInt OffsetInField = RestOff - FieldOff;
      auto &FieldType = Struct->Fields().at(FieldOff).Type();
      if (OffsetInField.uge(*FieldType.size(VH))) {
        return ModelGEPArgs{ .BaseAddress = TBA,
                             .IndexVector = std::move(Indices),
                             .RestOff = RestOff,
                             .PointeeType = FieldType };
      }

      // Then we subtract the field offset from the remaining offset
      RestOff = OffsetInField;
      CurrentType = FieldType;
    } break;

    case AggregateKind::Union: {
      revng_assert(CurrentType.is(model::TypeKind::UnionType));
      auto QualifiedUnion = peelConstAndTypedefs(CurrentType);
      const auto *Unqualified = QualifiedUnion.UnqualifiedType().getConst();
      const auto *Union = llvm::cast<model::UnionType>(Unqualified);

      // Index represents the number of the field in the union, this does not
      // affect the RestOff, since traversing union fields does not increase
      // the offset.
      revng_assert(not Back.InductionVariable);
      uint64_t FieldId = Back.ConstantIndex;
      auto &FieldType = Union->Fields().at(FieldId).Type();
      if (RestOff.uge(*FieldType.size(VH))) {
        return ModelGEPArgs{ .BaseAddress = TBA,
                             .IndexVector = std::move(Indices),
                             .RestOff = RestOff,
                             .PointeeType = FieldType };
      }

      CurrentType = FieldType;

    } break;

    default:
      revng_abort();
    }
  }

  revng_assert(RestOff.isNonNegative());
  return ModelGEPArgs{ .BaseAddress = TBA,
                       .IndexVector = std::move(Indices),
                       .RestOff = RestOff,
                       .PointeeType = CurrentType };
}

class GEPSummationCache {

  const model::Binary &Model;

  // This maps Uses to ModelGEPSummations so that in consecutive iterations on
  // consecutive instructions we can reuse parts of them without walking the
  // entire def-use chain.
  UseGEPSummationMap UseGEPSummations = {};

  RecursiveCoroutine<ModelGEPSummation>
  getGEPSumImpl(Use &AddressUse, const ModelTypesMap &PointerTypes) {
    revng_log(ModelGEPLog,
              "getGEPSumImpl for use of: " << dumpToString(AddressUse.get()));
    LoggerIndent Indent{ ModelGEPLog };

    ModelGEPSummation Result = {};

    // If it's already been handled, we already know if it can be
    // modelGEPified or not, so we stick to that decision.
    auto GEPItHint = UseGEPSummations.lower_bound(&AddressUse);
    if (GEPItHint != UseGEPSummations.end()
        and not(&AddressUse < GEPItHint->first)) {
      revng_log(ModelGEPLog, "Found!");
      rc_return GEPItHint->second;
    }
    revng_log(ModelGEPLog, "Not found. Compute one!");

    Value *AddressArith = AddressUse.get();
    // If the used value and we know it has a pointer type, we already know
    // both the base address and the pointer type.
    if (auto TypeIt = PointerTypes.find(AddressArith);
        TypeIt != PointerTypes.end()) {

      auto &[AddressVal, Type] = *TypeIt;

      revng_assert(Type.isPointer());
      revng_log(ModelGEPLog, "Use is typed!");

      Result = ModelGEPSummation{
        .BaseAddress = TypedBaseAddress{ .Type = dropPointer(Type),
                                         .Address = AddressArith },
        // The summation is empty since AddressArith
        // has exactly the type we're looking at here.
        .Summation = {}
      };

    } else if (isa<ConstantExpr>(AddressArith)) {
      revng_log(ModelGEPLog, "Traverse cast!");
      Result = makeOffsetGEPSummation(AddressArith);
    } else if (isa<Instruction>(AddressArith)) {

      auto *AddrArithmeticInst = dyn_cast<Instruction>(AddressArith);
      auto *ConstExprAddrArith = dyn_cast<ConstantExpr>(AddressArith);
      if (ConstExprAddrArith)
        AddrArithmeticInst = ConstExprAddrArith->getAsInstruction();

      switch (AddrArithmeticInst->getOpcode()) {

      case Instruction::Add: {
        auto *Add = cast<BinaryOperator>(AddrArithmeticInst);

        Use &LHSUse = Add->getOperandUse(0);
        auto LHSSummation = rc_recur getGEPSumImpl(LHSUse, PointerTypes);

        Use &RHSUse = Add->getOperandUse(1);
        auto RHSSummation = rc_recur getGEPSumImpl(RHSUse, PointerTypes);

        if (not RHSSummation.isValid() or not LHSSummation.isValid())
          break;

        bool LHSIsAddress = LHSSummation.isAddress();
        bool RHSIsAddress = RHSSummation.isAddress();
        // In principle we should not expect to have many base address.
        // If we do, at the moment we don't have a better policy than to bail
        // out, and in principle this is totally safe, even if we give up a
        // chance to emit good model geps for this case.
        // In any case, we might want to devise smarter policies to
        // discriminate between different base addresses. Anyway it's not
        // clear if we can ever do something better than this.
        if (LHSIsAddress and RHSIsAddress) {
          Result = ModelGEPSummation::invalid();
        } else {

          // If both LHS and RHS are not addresses (both are plain offset
          // arithmetic) or just one is an address (and the other is offset
          // arithmetic) we take the address (if present) as starting point,
          // and add up the two summations.
          Result = LHSIsAddress ? LHSSummation : RHSSummation;
          Result.Summation.append(LHSIsAddress ? RHSSummation.Summation :
                                                 LHSSummation.Summation);
        }
      } break;

      case Instruction::ZExt:
      case Instruction::IntToPtr:
      case Instruction::PtrToInt:
      case Instruction::BitCast: {
        // casts are traversed
        revng_log(ModelGEPLog, "Traverse cast!");
        Result = rc_recur getGEPSumImpl(AddrArithmeticInst->getOperandUse(0),
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
          Result = ModelGEPSummation{
            // The base address is unknown
            .BaseAddress = TypedBaseAddress{ .Type = {}, .Address = nullptr },
            // The summation has only one element, with a constant coefficient,
            // and the index is the current instructions.
            .Summation = { ModelGEPSummationElement{ .Coefficient = ConstOp,
                                                     .Index = OtherOp } }
          };
        } else {
          // In all the other cases, fall back to treating this as a
          // non-address and non-strided instruction, just like e.g. division.

          Result = makeOffsetGEPSummation(AddrArithmeticInst);
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

              Result = ModelGEPSummation{
                // The base address is unknown
                .BaseAddress = TypedBaseAddress{ .Type = {},
                                                 .Address = nullptr },
                // The summation has only one element, with a coefficient of 1,
                // and the index is the current instructions.
                .Summation = { ModelGEPSummationElement{
                  .Coefficient = Stride, .Index = IndexForStridedAccess } }
              };
              // Then we're done, break from the switch
              break;
            }
          }
        }

        // In all the other cases, fall back to treating this as a non-address
        // and non-strided instruction, just like e.g. division.

        Result = makeOffsetGEPSummation(AddrArithmeticInst);

      } break;

      case Instruction::Alloca: {
        Result = ModelGEPSummation::invalid();
      } break;

      case Instruction::GetElementPtr: {
        revng_abort("TODO: gep is not supported by make-model-gep yet");
      } break;

      case Instruction::Load:
      case Instruction::Call:
      case Instruction::PHI:
      case Instruction::ExtractValue:

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
        Result = makeOffsetGEPSummation(AddrArithmeticInst);
      } break;

      case Instruction::Unreachable:
      case Instruction::Store:
      case Instruction::InsertValue:
      case Instruction::Invoke:
      case Instruction::Resume:
      case Instruction::CleanupRet:
      case Instruction::CatchRet:
      case Instruction::CatchPad:
      case Instruction::CatchSwitch:
      case Instruction::AtomicCmpXchg:
      case Instruction::AtomicRMW:
      case Instruction::Fence:
      case Instruction::FAdd:
      case Instruction::FSub:
      case Instruction::FMul:
      case Instruction::FDiv:
      case Instruction::FRem:
      case Instruction::FPTrunc:
      case Instruction::FPExt:
      case Instruction::FCmp:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::UIToFP:
      case Instruction::SIToFP:
      case Instruction::AddrSpaceCast:
      case Instruction::VAArg:
      case Instruction::ExtractElement:
      case Instruction::InsertElement:
      case Instruction::ShuffleVector:
      case Instruction::LandingPad:
      case Instruction::CleanupPad:
      case Instruction::Br:
      case Instruction::IndirectBr:
      case Instruction::Ret:
      case Instruction::Switch: {
        revng_abort("unexpected instruction for address arithmetic");
      } break;

      default: {
        revng_abort("Unexpected operation");
      } break;
      }

      if (ConstExprAddrArith)
        AddrArithmeticInst->deleteValue();

    } else if (auto *Const = dyn_cast<ConstantInt>(AddressArith)) {

      // If we reach this point the constant int does not represent a pointer
      // so we initialize the result as if it was an offset
      if (Const->getValue().isNonNegative())
        Result = makeOffsetGEPSummation(Const);

    } else if (auto *Arg = dyn_cast<Argument>(AddressArith)) {

      // If we reach this point the argument does not represent a pointer so
      // we initialize the result as if it was an offset
      Result = makeOffsetGEPSummation(Arg);

    } else if (isa<GlobalVariable>(AddressArith)
               or isa<llvm::UndefValue>(AddressArith)
               or isa<llvm::PoisonValue>(AddressArith)) {

      Result = ModelGEPSummation::invalid();

    } else {
      // We don't expect other stuff. This abort is mainly intended to be a
      // safety net during development. It can eventually be dropped.
      AddressArith->dump();
      revng_abort();
    }

    UseGEPSummations.insert(GEPItHint, { &AddressUse, Result });

    rc_return Result;
  }

  ModelGEPSummation makeOffsetGEPSummation(Value *V) const {

    // If we reach one of these instructions, it definitely cannot be an
    // address, but it's just considered as regular offset arithmetic of an
    // unknown offset.
    auto *VType = V->getType();
    auto *ArithTy = dyn_cast<llvm::IntegerType>(VType);
    if (not ArithTy) {
      // If we're dealing with something whose type is a pointer, it cannot be
      // an offset. So return an invalid ModelGEPSummation.
      return ModelGEPSummation::invalid();
    }

    using model::Architecture::getPointerSize;
    size_t PointerBytes = getPointerSize(Model.Architecture());
    APInt TheOne = APInt(/*NumBits*/ 8 * PointerBytes, /*Value*/ 1);
    auto *One = ConstantInt::get(V->getContext(), TheOne);
    return ModelGEPSummation{
      // The base address is unknown
      .BaseAddress = TypedBaseAddress{ .Type = {}, .Address = nullptr },
      // The summation has only one element, with a coefficient of 1, and
      // the
      // index is the current instructions.
      .Summation = { ModelGEPSummationElement{ .Coefficient = One,
                                               .Index = V } }
    };
  }

public:
  GEPSummationCache(const model::Binary &M) : Model(M), UseGEPSummations() {}

  void clear() { UseGEPSummations.clear(); }

  ModelGEPSummation
  getGEPSummation(Use &AddressUse, const ModelTypesMap &PointerTypes) {
    return getGEPSumImpl(AddressUse, PointerTypes);
  }
};

using UseGEPInfoMap = std::map<Use *, ModelGEPArgs>;

static UseGEPInfoMap makeGEPReplacements(llvm::Function &F,
                                         const model::Binary &Model,
                                         model::VerifyHelper &VH,
                                         FunctionMetadataCache &Cache) {

  UseGEPInfoMap Result;

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

  GEPSummationCache GEPSumCache{ Model };
  // TypedAccessCache TAPCache;

  UseTypeMap GEPifiedUsedTypes;

  // LLVMContext &Ctxt = F.getContext();
  auto RPOT = ReversePostOrderTraversal(&F.getEntryBlock());
  for (auto *BB : RPOT) {
    for (auto &I : *BB) {
      revng_log(ModelGEPLog, "Instruction " << dumpToString(&I));
      auto Indent = LoggerIndent{ ModelGEPLog };

      if (auto *CallI = dyn_cast<CallInst>(&I)) {
        if (not FunctionTags::CallToLifted.isTagOf(CallI)) {
          revng_log(ModelGEPLog, "Skipping call to non-isolated function");
          continue;
        }
      }

      for (Use &U : I.operands()) {

        // Skip BasicBlocks, they cannot be arithmetic
        if (isa<llvm::BasicBlock>(U.get())) {
          revng_log(ModelGEPLog, "Skipping basic block operand");
          continue;
        }

        // Skip callee operands in CallInst if it's not an llvm::Instruction.
        // If it is an Instruction we should handle it.
        if (auto *CallUser = dyn_cast<CallInst>(U.getUser())) {
          if (not isa<llvm::Instruction>(CallUser->getCalledOperand())) {
            if (&U == &CallUser->getCalledOperandUse()) {
              revng_log(ModelGEPLog, "Skipping callee operand in CallInst");
              continue;
            }
          }
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

        // Skip null pointer constants, and undefs, since they cannot be valid
        // addresses
        // TODO: if we ever need to support memory mapped at address 0 we can
        // probably work around this, but this is not top priority for now.
        if (isa<llvm::UndefValue>(U.get()) or isa<llvm::PoisonValue>(U.get())
            or isa<llvm::ConstantPointerNull>(U.get())) {
          revng_log(ModelGEPLog, "Skipping null pointer address");
          continue;
        }

        ModelGEPSummation GEPSum = GEPSumCache.getGEPSummation(U, PointerTypes);

        revng_log(ModelGEPLog, "GEPSum " << GEPSum);
        if (not GEPSum.isAddress())
          continue;

        // Pre-compute all the typed access patterns from the base address of
        // the GEPSum, or get them from the caches if we've already computed
        // them.
        const model::QualifiedType &BaseTy = GEPSum.BaseAddress.Type;

        // If the base type is a funcion type we have nothing to do, because
        // function types cannot be "traversed" with ModelGEP.
        if (BaseTy.is(model::TypeKind::RawFunctionType)
            or BaseTy.is(model::TypeKind::CABIFunctionType))
          continue;

        // Now we extract an IRAccessPattern from the ModelGEPSummation
        IRAccessPattern IRPattern = computeIRAccessPattern(Cache,
                                                           U,
                                                           GEPSum,
                                                           Model,
                                                           PointerTypes,
                                                           GEPifiedUsedTypes);

        // Select among the computed TAPIndices the one which best fits the
        // IRPattern
        ModelGEPArgs GEPArgs = makeBestGEPArgs(GEPSum.BaseAddress,
                                               IRPattern,
                                               Model,
                                               VH);

        const model::Architecture::Values &Architecture = Model.Architecture();
        auto PointerToGEPArgs = GEPArgs.PointeeType.getPointerTo(Architecture);
        GEPifiedUsedTypes.insert({ &U, PointerToGEPArgs });

        revng_log(ModelGEPLog, "Best GEPArgs: " << GEPArgs);

        // If GEPSum is an address and I is an "address barrier"
        // instruction (e.g. an instruction such that pointer arithmetics does
        // not propagate through it), we need to check if we can still deduce
        // a rich pointer type for I starting from GEPSum. An example of an
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
        if (auto *Load = dyn_cast<LoadInst>(&I)) {
          std::optional<model::QualifiedType>
            GEPTypeOrNone = getType(GEPArgs.BaseAddress.Type,
                                    GEPArgs.IndexVector,
                                    GEPArgs.RestOff,
                                    VH);
          if (GEPTypeOrNone.has_value()) {
            model::QualifiedType &GEPType = GEPTypeOrNone.value();
            if (GEPType.isPointer())
              PointerTypes.insert({ Load, GEPType });
          }
        }

        Result[&U] = GEPArgs;
      }
    }
  }
  return Result;
}

class ModelGEPArgCache {

  std::map<model::QualifiedType, Constant *> GlobalModelGEPTypeArgs;

public:
  Value *getQualifiedTypeArg(model::QualifiedType &QT, llvm::Module &M) {
    auto It = GlobalModelGEPTypeArgs.find(QT);
    if (It != GlobalModelGEPTypeArgs.end())
      return It->second;

    It = GlobalModelGEPTypeArgs.insert({ QT, serializeToLLVMString(QT, M) })
           .first;
    return It->second;
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
  UseGEPInfoMap GEPReplacementMap = makeGEPReplacements(F, *Model, VH, Cache);

  llvm::Module &M = *F.getParent();
  LLVMContext &Ctxt = M.getContext();
  IRBuilder<> Builder(Ctxt);
  ModelGEPArgCache TypeArgCache;

  // Create a function pool for AddressOf calls
  OpaqueFunctionsPool<TypePair> AddressOfPool(&M,
                                              /* PurgeOnDestruction */ false);
  if (not GEPReplacementMap.empty())
    initAddressOfPool(AddressOfPool, &M);

  llvm::IntegerType *PtrSizedInteger = getPointerSizedInteger(Ctxt, *Model);

  for (auto &[TheUseToGEPify, GEPArgs] : GEPReplacementMap) {

    // Skip ModelGEPs that have no arguments
    if (GEPArgs.IndexVector.empty())
      continue;

    revng_log(ModelGEPLog,
              "GEPify use of: " << dumpToString(TheUseToGEPify->get()));
    revng_log(ModelGEPLog,
              "  `-> use in: " << dumpToString(TheUseToGEPify->getUser()));

    llvm::Type *UseType = TheUseToGEPify->get()->getType();
    llvm::Type *BaseAddrType = GEPArgs.BaseAddress.Address->getType();

    llvm::IntegerType *ModelGEPReturnedType;

    // Calculate the size of the GEPPed field
    if (GEPArgs.RestOff.isStrictlyPositive()) {
      // If there is a remaining offset, we are returning something more
      // similar to a pointer than the actual value
      ModelGEPReturnedType = PtrSizedInteger;
    } else {
      std::optional<uint64_t> PointeeSize = GEPArgs.PointeeType.size(VH);
      revng_assert(PointeeSize.has_value());

      ModelGEPReturnedType = llvm::IntegerType::get(Ctxt,
                                                    PointeeSize.value() * 8);
    }

    auto *ModelGEPFunction = getModelGEP(M, ModelGEPReturnedType, BaseAddrType);

    // Build the arguments for the call to modelGEP
    SmallVector<Value *, 4> Args;
    Args.reserve(GEPArgs.IndexVector.size() + 2);

    // The first argument is always a pointer to a constant global variable
    // that holds the string representing the yaml serialization of the
    // qualified type of the base type of the modelGEP
    model::QualifiedType &BaseType = GEPArgs.BaseAddress.Type;
    auto *BaseTypeConstantStrPtr = TypeArgCache.getQualifiedTypeArg(BaseType,
                                                                    M);
    Args.push_back(BaseTypeConstantStrPtr);

    // The second argument is the base address
    Args.push_back(GEPArgs.BaseAddress.Address);

    auto *UserInstr = cast<Instruction>(TheUseToGEPify->getUser());
    if (auto *PHIUser = dyn_cast<PHINode>(UserInstr)) {
      auto *IncomingB = PHIUser->getIncomingBlock(*TheUseToGEPify);
      Builder.SetInsertPoint(IncomingB->getTerminator());
    } else {
      Builder.SetInsertPoint(UserInstr);
    }

    // The other arguments are the indices in IndexVector
    for (auto [ConstantIndex, InductionVariable, AggregateTy] :
         GEPArgs.IndexVector) {

      if (InductionVariable) {
        revng_assert(AggregateTy == AggregateKind::Array);
        if (ConstantIndex) {
          auto *FixedId = llvm::ConstantInt::get(InductionVariable->getType(),
                                                 ConstantIndex);
          Args.push_back(Builder.CreateAdd(InductionVariable, FixedId));
        } else {
          Args.push_back(InductionVariable);
        }

      } else {
        auto *Int64Type = llvm::IntegerType::get(Ctxt, 64 /*NumBits*/);
        auto *FixedId = llvm::ConstantInt::get(Int64Type, ConstantIndex);
        Args.push_back(FixedId);
      }
    }

    Value *ModelGEPRef = Builder.CreateCall(ModelGEPFunction, Args);

    auto AddrOfReturnedType = UseType;
    if (GEPArgs.RestOff.isStrictlyPositive()) {
      // If there is a remaining offset, we are returning something more
      // similar to a pointer than the actual value
      AddrOfReturnedType = PtrSizedInteger;
    }

    // Inject a call to AddressOf
    auto *AddressOfFunctionType = getAddressOfType(AddrOfReturnedType,
                                                   ModelGEPReturnedType);
    auto *AddressOfFunction = AddressOfPool.get({ AddrOfReturnedType,
                                                  ModelGEPReturnedType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    auto *PointeeConstantStrPtr = TypeArgCache
                                    .getQualifiedTypeArg(GEPArgs.PointeeType,
                                                         M);
    Value *ModelGEPPtr = Builder.CreateCall(AddressOfFunction,
                                            { PointeeConstantStrPtr,
                                              ModelGEPRef });

    if (GEPArgs.RestOff.isStrictlyPositive()) {
      // If the GEPArgs have a RestOff that is strictly positive, we have to
      // inject the remaining part of the pointer arithmetic as normal sums
      auto GEPResultBitWidth = ModelGEPPtr->getType()->getIntegerBitWidth();
      APInt OffsetToAdd = GEPArgs.RestOff.zextOrTrunc(GEPResultBitWidth);
      ModelGEPPtr = Builder.CreateAdd(ModelGEPPtr,
                                      ConstantInt::get(Ctxt, OffsetToAdd));

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

    Changed = true;
  }

  if (VerifyLog.isEnabled())
    revng_assert(not llvm::verifyModule(*F.getParent(), &llvm::dbgs()));

  return Changed;
}

char MakeModelGEPPass::ID = 0;

using Pass = MakeModelGEPPass;
static RegisterPass<Pass> X("make-model-gep",
                            "Pass that transforms address arithmetic into "
                            "calls to ModelGEP ",
                            false,
                            false);
