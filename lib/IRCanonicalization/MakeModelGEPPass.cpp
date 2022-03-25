//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
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

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

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
using llvm::ExtractValueInst;
using llvm::FunctionPass;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::IRBuilder;
using llvm::isa;
using llvm::LLVMContext;
using llvm::LoadInst;
using llvm::None;
using llvm::Optional;
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

static Logger<> ModelGEPLog{ "make-model-gep" };

struct MakeModelGEPPass : public FunctionPass {
public:
  static char ID;

  MakeModelGEPPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }
};

using ValueModelTypesMap = std::map<const Value *, const model::QualifiedType>;

ValueModelTypesMap initializeModelTypes(const llvm::Function &F,
                                        const model::Function &ModelF,
                                        const model::Binary &Model) {
  ValueModelTypesMap Result;

  auto Indent = LoggerIndent(ModelGEPLog);

  const model::Type *FType = ModelF.Prototype.get();
  revng_assert(FType);

  // TODO we should create types for ConstantInts that are valid addresses that
  // point into segments

  // First, initialize the types of F's arguments
  revng_log(ModelGEPLog, "Initialize argument types");
  if (const auto *RFT = dyn_cast<model::RawFunctionType>(FType)) {

    auto MoreIndent = LoggerIndent(ModelGEPLog);
    revng_log(ModelGEPLog, "model::RawFunctionType");
    unsigned ActualArgSize = RFT->Arguments.size()
                             + (RFT->StackArgumentsType.isValid() ? 1 : 0);
    revng_assert(F.arg_size() == ActualArgSize);

    auto MoreMoreIndent = LoggerIndent(ModelGEPLog);
    for (const auto &[ModelArg, LLVMArg] :
         llvm::zip_first(RFT->Arguments, F.args())) {
      auto _ = LoggerIndent(ModelGEPLog);
      revng_log(ModelGEPLog, "llvm::Argument: " << dumpToString(LLVMArg));
      revng_log(ModelGEPLog,
                "model::QualifiedType: " << serializeToString(ModelArg.Type));
      if (ModelArg.Type.isPointer()) {
        revng_log(ModelGEPLog, "INITIALIZED");
        Result.insert({ &LLVMArg, ModelArg.Type });
      }
    }
  } else if (const auto *CFT = dyn_cast<model::CABIFunctionType>(FType)) {

    auto MoreIndent = LoggerIndent(ModelGEPLog);
    revng_assert(CFT->Arguments.size() == F.arg_size());
    revng_log(ModelGEPLog, "model::CABIFunctionType");

    auto MoreMoreIndent = LoggerIndent(ModelGEPLog);
    for (const auto &[ModelArg, LLVMArg] :
         llvm::zip_first(CFT->Arguments, F.args())) {
      auto _ = LoggerIndent(ModelGEPLog);
      revng_log(ModelGEPLog, "llvm::Argument: " << dumpToString(LLVMArg));
      revng_log(ModelGEPLog,
                "model::QualifiedType: " << serializeToString(ModelArg.Type));
      if (ModelArg.Type.isPointer()) {
        revng_log(ModelGEPLog, "INITIALIZED");
        Result.insert({ &LLVMArg, ModelArg.Type });
      }
    }
  } else {
    revng_abort("Function should have RawFunctionType or CABIFunctionType");
  }

  for (auto &I : llvm::instructions(F)) {
    auto MoreIndent = LoggerIndent(ModelGEPLog);
    revng_log(ModelGEPLog, "Instruction " << dumpToString(&I));
    auto MoreMoreIndent = LoggerIndent(ModelGEPLog);

    // For calls we have some cases we want to catch:
    // - return values for which we have types on the model
    // - special functions that initialize stack-allocated stuff (stack
    //   variables, stack arguments passed to call sites)
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      revng_log(ModelGEPLog, "Call");

      // Special case for calls to special functions that initialize
      // stack-allocated stuff
      auto *Callee = Call->getCalledFunction();
      if (Callee) {
        if (Callee->getName() == "revng_stack_frame") {
          if (ModelF.StackFrameType.isValid()) {

            auto PointerQual = Qualifier::createPointer(Model.Architecture);
            model::QualifiedType FStackType(ModelF.StackFrameType,
                                            { PointerQual });
            revng_log(ModelGEPLog, "Call: " << dumpToString(Call));
            revng_log(ModelGEPLog,
                      "model::QualifiedType: "
                        << serializeToString(FStackType));
            revng_log(ModelGEPLog, "INITIALIZED");
            Result.insert({ Call, std::move(FStackType) });
          }

          continue;

        } else if (Callee->getName() == "revng_call_stack_arguments") {

          for (const Use &StackArgsUse : Call->uses()) {
            auto *CallUsingArgs = dyn_cast<CallInst>(StackArgsUse.getUser());
            if (CallUsingArgs) {
              // The stack argument should be the last
              unsigned NArgOperands = CallUsingArgs->getNumArgOperands();
              revng_assert(StackArgsUse.getOperandNo() == NArgOperands - 1);

              const model::Type *CalleeT = getCallSitePrototype(Model, Call);
              revng_assert(CalleeT);
              const auto *CalleeRFT = cast<model::RawFunctionType>(CalleeT);
              revng_assert(CalleeRFT->StackArgumentsType.isValid());

              auto PointerQual = Qualifier::createPointer(Model.Architecture);
              model::QualifiedType
                CalleeStackType(CalleeRFT->StackArgumentsType, { PointerQual });
              Result.insert({ Call, std::move(CalleeStackType) });
            }
          }
          continue;
        }
      }

      // If we reach this point, the call is not calling a special-cased
      // stack-allocation function, so we need to check if it calls an
      // isolated function.

      // If this call does not have a prototype we have no types to inizialize.
      // Just go on with the next instruction.
      const model::Type *FType = getCallSitePrototype(Model, Call);
      if (not FType) {
        revng_log(ModelGEPLog,
                  "Could not retrieve the prototype. Skipping ...");
        continue;
      }

      if (const auto *RFT = dyn_cast<model::RawFunctionType>(FType)) {
        revng_log(ModelGEPLog, "Call has RawFunctionType prototype.");

        // If the callee function does not return anything, skip to the next
        // instruction.
        if (RFT->ReturnValues.empty()) {
          revng_log(ModelGEPLog, "Does not return values on model. Skip ...");
          revng_assert(Call->getType()->isVoidTy());
          continue;
        }

        if (RFT->ReturnValues.size() == 1) {
          revng_log(ModelGEPLog, "Has single return type.");

          revng_assert(Call->getType()->isVoidTy()
                       or Call->getType()->isIntOrPtrTy());

          const model::QualifiedType &ModT = RFT->ReturnValues.begin()->Type;
          if (ModT.isPointer()) {
            auto _ = LoggerIndent(ModelGEPLog);
            revng_log(ModelGEPLog, "llvm::CallInst: " << dumpToString(Call));
            revng_log(ModelGEPLog,
                      "model::QualifiedType: " << serializeToString(ModT));
            Result.insert({ Call, ModT });
          }

        } else {
          auto *StrucT = cast<llvm::StructType>(Call->getType());
          revng_log(ModelGEPLog, "Has many return types.");
          revng_assert(StrucT->getNumElements() == RFT->ReturnValues.size());

          if (not Call->getNumUses()) {
            revng_log(ModelGEPLog, "Has no uses. Skip ...");
            continue;
          }

          const auto Extracted = getExtractedValuesFromInstruction(Call);
          revng_assert(Extracted.size() == StrucT->getNumElements());
          for (const auto &[ReturnValue, ExtractedSet] :
               llvm::zip_first(RFT->ReturnValues, Extracted)) {
            // Inside here we're working on a signle field of the struct.
            // ExtractedSet contains all the ExtractValueInst that extract the
            // same field of the struct.
            const model::QualifiedType &ModT = ReturnValue.Type;
            if (ModT.isPointer()) {
              for (auto *V : ExtractedSet) {
                revng_assert(isa<ExtractValueInst>(V));
                auto _ = LoggerIndent(ModelGEPLog);
                revng_log(ModelGEPLog,
                          "llvm::ExtractValueInst: " << dumpToString(V));
                revng_log(ModelGEPLog,
                          "model::QualifiedType: " << serializeToString(ModT));
                Result.insert({ V, ModT });
              }
            }
          }
        }

      } else if (const auto *CFT = dyn_cast<model::CABIFunctionType>(FType)) {
        revng_log(ModelGEPLog, "Call has CABIFunctionType prototype.");

        // If the callee function does not return anything, skip to the next
        // instruction.
        if (CFT->ReturnType.isVoid()) {
          revng_log(ModelGEPLog, "Returns void. Skip ...");
          revng_assert(Call->getType()->isVoidTy());
          continue;
        }

        // TODO: we haven't handled how to emit return values of CABIFunctions
        revng_log(ModelGEPLog, "CABIFunctionType found, ignoring.");

      } else {
        revng_abort("Function should have RawFunctionType or "
                    "CABIFunctionType");
      }
    }
  }

  revng_log(ModelGEPLog, "Done initializing argument types");

  return Result;
}

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
      // The summation has only one element, which is not valid, because it does
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
  Optional<model::QualifiedType> PointeeType = None;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "IRAccessPattern {\nBaseOffset: " << BaseOffset.toString(10, true)
       << "\nIndices = {";
    for (const auto &I : Indices) {
      OS << "\n";
      I.dump(OS);
    }
    OS << "}\nPointeeType: ";
    if (PointeeType.hasValue())
      serialize(OS, PointeeType.getValue());
    else
      OS << "std::nullopt";
    OS << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

static IRAccessPattern computeAccessPattern(const Use &U,
                                            const ModelGEPSummation &GEPSum,
                                            const model::Binary &Model) {
  revng_assert(GEPSum.isAddress());

  // First, prepare the BaseOffset and the Indices for the IRAccessPattern.
  GEPSummationVector IRPatternIndices;
  size_t PointerBytes = model::Architecture::getPointerSize(Model.Architecture);
  APInt BaseOff = APInt(/*NumBits*/ 8 * PointerBytes, /*Value*/ 0);

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
  IRAccessPattern IRPattern{ .BaseOffset = BaseOff,
                             .Indices = IRPatternIndices,
                             // Intially PointeeType is set to None, then we
                             // fill it if in some special cases where we have
                             // interesting information on the pointee
                             .PointeeType = None };

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

      model::TypePath
        Pointee = Model.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                         PointeeSize);
      model::QualifiedType QPointee = model::QualifiedType(Pointee, {});
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
        model::QualifiedType QPointee = model::QualifiedType(Pointee, {});
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

      const model::Type *FType = MF->Prototype.get();
      revng_assert(FType);

      if (const auto *RFT = dyn_cast<model::RawFunctionType>(FType)) {
        revng_log(ModelGEPLog, "Has RawFunctionType prototype.");

        // If the callee function does not return anything, skip to the next
        // instruction.
        if (RFT->ReturnValues.empty()) {
          revng_log(ModelGEPLog, "Does not return values on model. Skip ...");
          revng_assert(not Ret->getReturnValue());
        } else if (RFT->ReturnValues.size() == 1) {
          revng_log(ModelGEPLog, "Has single return type.");

          revng_assert(Ret->getReturnValue()->getType()->isVoidTy()
                       or Ret->getReturnValue()->getType()->isIntOrPtrTy());

          const model::QualifiedType &ModT = RFT->ReturnValues.begin()->Type;
          // If the returned type is a pointer, we unwrap it and set the pointee
          // type of IRPattern to the pointee of the return type.
          // Otherwise the Function is not returning a pointer, and we can skip
          // it.
          if (ModT.isPointer()) {
            auto _ = LoggerIndent(ModelGEPLog);
            revng_log(ModelGEPLog, "llvm::ReturnInst: " << dumpToString(Ret));
            revng_log(ModelGEPLog,
                      "Pointee: model::QualifiedType: "
                        << serializeToString(ModT));
            IRPattern.PointeeType = dropPointer(ModT);
          }

        } else {
          auto *RetVal = Ret->getReturnValue();
          auto *StructTy = cast<llvm::StructType>(RetVal->getType());
          revng_log(ModelGEPLog, "Has many return types.");
          revng_assert(StructTy->getNumElements() == RFT->ReturnValues.size());

          // Assert that we're returning a proper struct, initialized with
          // struct initializers, but don't do anything here.
          const auto *Returned = cast<CallInst>(RetVal)->getCalledFunction();
          revng_assert(FunctionTags::StructInitializer.isTagOf(Returned));
        }

      } else if (const auto *CFT = dyn_cast<model::CABIFunctionType>(FType)) {
        revng_log(ModelGEPLog, "Has CABIFunctionType prototype.");

        // If the callee function does not return anything, skip to the next
        // instruction.
        if (CFT->ReturnType.isVoid()) {
          revng_log(ModelGEPLog, "Returns void. Skip ...");
          revng_assert(not Ret->getReturnValue());
        } else {

          // TODO: we haven't handled return values of CABIFunctions yet
          revng_abort();
        }

      } else {
        revng_abort("Function should have RawFunctionType or "
                    "CABIFunctionType");
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
        const model::Type *CalledPrototype = CalledFType->Prototype.getConst();

        if (auto *RFT = dyn_cast<RawFunctionType>(CalledPrototype)) {
          revng_log(ModelGEPLog, "Has RawFunctionType prototype.");
          revng_assert(RFT->ReturnValues.size() > 1);

          auto *StructTy = cast<llvm::StructType>(CalledF->getReturnType());
          revng_log(ModelGEPLog, "Has many return types.");
          revng_assert(StructTy->getNumElements() == RFT->ReturnValues.size());

          model::QualifiedType
            RetTy = std::next(RFT->ReturnValues.begin(), ArgNum)->Type;
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
        const model::Type *FType = getCallSitePrototype(Model, Call);

        if (const auto *RFT = dyn_cast<model::RawFunctionType>(FType)) {

          auto MoreIndent = LoggerIndent(ModelGEPLog);
          auto ModelArgSize = RFT->Arguments.size();
          revng_assert(ModelArgSize == Call->arg_size()
                       or (ModelArgSize == Call->arg_size() - 1
                           and RFT->StackArgumentsType.isValid()));
          revng_log(ModelGEPLog, "model::RawFunctionType");

          auto _ = LoggerIndent(ModelGEPLog);
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
            ArgTy.UnqualifiedType = RFT->StackArgumentsType;
            revng_assert(ArgTy.UnqualifiedType.isValid());
          } else {
            auto ArgIt = std::next(RFT->Arguments.begin(), ArgOpNum);
            ArgTy = ArgIt->Type;
          }

          revng_log(ModelGEPLog,
                    "model::QualifiedType: " << serializeToString(ArgTy));
          if (ArgTy.isPointer()) {
            model::QualifiedType Pointee = dropPointer(ArgTy);
            revng_log(ModelGEPLog, "Pointee: " << serializeToString(Pointee));
            IRPattern.PointeeType = Pointee;
          }

        } else if (const auto *CFT = dyn_cast<model::CABIFunctionType>(FType)) {

          auto MoreIndent = LoggerIndent(ModelGEPLog);
          revng_assert(CFT->Arguments.size() == Call->arg_size());
          revng_log(ModelGEPLog, "model::CABIFunctionType");

          auto _ = LoggerIndent(ModelGEPLog);
          unsigned ArgOpNum = Call->getArgOperandNo(&U);
          revng_log(ModelGEPLog, "ArgOpNum: " << ArgOpNum);
          revng_log(ModelGEPLog, "ArgOperand: " << U.get());
          model::QualifiedType ArgTy = CFT->Arguments.at(ArgOpNum).Type;
          revng_log(ModelGEPLog,
                    "model::QualifiedType: " << serializeToString(ArgTy));
          if (ArgTy.isPointer()) {
            model::QualifiedType Pointee = dropPointer(ArgTy);
            revng_log(ModelGEPLog, "Pointee: " << serializeToString(Pointee));
            IRPattern.PointeeType = Pointee;
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

static bool compareQualifiedTypes(const model::QualifiedType &LHS,
                                  const model::QualifiedType &RHS) {
  if (LHS.Qualifiers < RHS.Qualifiers)
    return true;

  if (LHS.UnqualifiedType.get() < RHS.UnqualifiedType.get())
    return true;

  return false;
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

    return compareQualifiedTypes(AccessedType, Other.AccessedType);
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

enum AggregateKind { Struct, Union, Array };

std::string toString(AggregateKind K) {
  switch (K) {
  case Struct:
    return "Struct";
  case Union:
    return "Union";
  case Array:
    return "Array";
  }
  return "Invalid";
}

struct ChildInfo {
  Value *Index;
  AggregateKind Type;

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "ModelGEPSummationElement{\nIndex:\n";
    if (Index)
      Index->print(OS);
    else
      OS << "nullptr";
    OS << "\nType: " << toString(Type) << "\n}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

using ChildIndexVector = SmallVector<ChildInfo, 4>;

using TAPToChildIdsMap = std::map<TypedAccessPattern, ChildIndexVector>;

using QTLess = std::integral_constant<decltype(&compareQualifiedTypes),
                                      compareQualifiedTypes>;

// clang-format off
using QualifiedTypeToTAPChildIdsMap = std::map<const model::QualifiedType,
                                               TAPToChildIdsMap,
                                               QTLess>;
// clang-format on

using TAPToChildIdsMapConstRef = std::reference_wrapper<const TAPToChildIdsMap>;

struct DifferenceScore {
  // Higher Difference are for stuff that is farther apart from a perfect match.
  // 0 or lower scores are for accesses that insist exactly on the beginning of
  // the type. Negative scores are for accesses that insist exactly on the
  // beginning of the type, but allowing for various levels of customization
  // (such as e.g. accesses that go deeper inside the type system and perfect
  // match have a score that is "more negative").
  ssize_t Difference = std::numeric_limits<ssize_t>::max();

  // This field represents how deep the type system was traversed to compute the
  // score. Scores with a higher depth are considered better (so lower
  // difference) because it means that the type system was traversed deeply.
  size_t Depth = std::numeric_limits<size_t>::min();

  // Boolean to mark out-of-range accesses
  bool InRange = false;

  std::strong_ordering operator<=>(const DifferenceScore &Other) const {
    if (InRange != Other.InRange)
      return InRange ? std::strong_ordering::less :
                       std::strong_ordering::greater;
    auto Cmp = Difference <=> Other.Difference;
    if (Cmp != 0)
      return Cmp;

    // Notice that in the following line the terms are inverted, because lower
    // depth needs to be scored "better" (so lower difference).
    return Other.Depth <=> Depth;
  }

  void dump(llvm::raw_ostream &OS) const debug_function {
    OS << "DifferenceScore { .Difference = " << Difference
       << ", .Depth = " << Depth
       << ", .InRange = " << (InRange ? "true" : "false") << "}";
  }

  void dump() const debug_function { dump(llvm::dbgs()); }
};

struct ScoredIndices {
  // The score is optional, nullopt means that the difference score is infinity
  std::optional<DifferenceScore> Score = std::nullopt;
  ChildIndexVector Indices = {};

  static ScoredIndices invalid() { return ScoredIndices{}; }

  static ScoredIndices outOfBound(ssize_t DiffScore) {
    return ScoredIndices{ .Score = DifferenceScore{ .Difference = DiffScore,
                                                    .Depth = 0,
                                                    .InRange = false },
                          .Indices{} };
  }
  static ScoredIndices nestedOutOfBound(ssize_t DiffScore,
                                        size_t Depth,
                                        ChildIndexVector &&Indices) {
    revng_assert(Depth <= Indices.size());
    if (Depth < Indices.size())
      Indices.resize(Depth);
    return ScoredIndices{ .Score = DifferenceScore{ .Difference = DiffScore,
                                                    .Depth = Depth,
                                                    .InRange = false },
                          .Indices = std::move(Indices) };
  }
};

static ScoredIndices
differenceScore(const model::QualifiedType &BaseType,
                const TAPToChildIdsMap::value_type &TAPWithIndices,
                const IRAccessPattern &IRAP,
                model::VerifyHelper &VH) {

  auto Result = ScoredIndices::outOfBound(IRAP.BaseOffset.getSExtValue());

  const auto &[TAP, ChildIndices] = TAPWithIndices;
  ChildIndexVector ResultIndices = ChildIndices;

  size_t BaseSize = *BaseType.size();
  if (IRAP.BaseOffset.uge(BaseSize))
    return Result;

  revng_assert(TAP.BaseOffset.ult(BaseSize));
  revng_assert((TAP.BaseOffset + *TAP.AccessedType.size()).ule(BaseSize));

  APInt RestOff = IRAP.BaseOffset;

  auto ArrayInfoIt = TAP.Arrays.begin();
  auto ArrayInfoEnd = TAP.Arrays.end();

  auto IRAPIndicesIt = IRAP.Indices.begin();
  auto IRAPIndicesEnd = IRAP.Indices.end();

  model::QualifiedType NestedType = BaseType;

  for (auto &ChildID : ResultIndices) {
    model::QualifiedType Normalized = peelConstAndTypedefs(NestedType, VH);

    // Should not be a pointer, because pointers don't have children on the
    // type system, which means that we shouldn't have a ChildId at this
    // point.
    revng_assert(not Normalized.isPointer());

    switch (ChildID.Type) {

    case Struct: {
      revng_assert(not Normalized.isArray());

      auto *S = cast<model::StructType>(Normalized.UnqualifiedType.get());
      size_t FieldOffset = cast<ConstantInt>(ChildID.Index)->getZExtValue();

      // If the RestOff is less than the field offset, it means that the IRAP
      // does not have enough offset to reach the field of the struct that is
      // required from the TAPWithIndices.
      // So we just bail out.
      if (RestOff.ult(FieldOffset))
        return ScoredIndices::invalid();

      RestOff -= FieldOffset;

      NestedType = S->Fields.at(FieldOffset).Type;
      ++Result.Score->Depth;
    } break;

    case Union: {
      revng_assert(not Normalized.isArray());
      auto *U = cast<model::UnionType>(Normalized.UnqualifiedType.get());
      size_t FieldID = cast<ConstantInt>(ChildID.Index)->getZExtValue();
      NestedType = U->Fields.at(FieldID).Type;
      ++Result.Score->Depth;
    } break;

    case Array: {
      revng_assert(Normalized.isArray());
      revng_assert(ArrayInfoIt != ArrayInfoEnd);

      const auto ArrayQualEnd = Normalized.Qualifiers.end();
      const auto ArrayQualIt = llvm::find_if(Normalized.Qualifiers,
                                             model::Qualifier::isArray);
      revng_assert(ArrayQualIt != ArrayQualEnd);

      revng_assert(not ChildID.Index);
      if (IRAPIndicesIt == IRAPIndicesEnd) {
        // This means that the IRAP does not have strided accesses anymore.
        // Hence for performing this array access it's using a constant offset
        // that needs to be translated into an index into the array.

        // The sizes of the array should be equal, but ArrayInfoIt->NumElems
        // could be 1 less than necessary because of some workarounds we have
        // built in DLA to handle arrays.
        revng_assert(ArrayQualIt->Size == ArrayInfoIt->NumElems
                     or (ArrayQualIt->Size - 1 == ArrayInfoIt->NumElems));

        APInt ElemIndex;
        APInt OffInElem;
        APInt::udivrem(RestOff, ArrayInfoIt->Stride, ElemIndex, OffInElem);

        if (ElemIndex.uge(ArrayInfoIt->NumElems)) {
          // If IRAP is trying to access an element that is larger than the
          // array size, we have to bail out, marking this as out of bound.
          return ScoredIndices::nestedOutOfBound(RestOff.getSExtValue(),
                                                 Result.Score->Depth,
                                                 std::move(ResultIndices));
        }

        RestOff = OffInElem;

      } else {
        revng_assert(not isa<ConstantInt>(IRAPIndicesIt->Index));

        // If the IRAccessPattern has a coefficient that is different from the
        // Stride of ArrayInfo, it means that IRAPIndicesIt->Index is not the
        // index of an element in the array, so we bail out.
        if (IRAPIndicesIt->Coefficient->getZExtValue() != ArrayInfoIt->Stride) {
          // TODO: in principle we could score this guy for similarity anyway.
          // But we need to return an llvm::Value or something that represents
          // what's left to add to the modelGEP if we select this.
          return ScoredIndices::invalid();
        }

        ++IRAPIndicesIt;
        ++Result.Score->Depth;
      }

      NestedType = model::QualifiedType(Normalized.UnqualifiedType,
                                        { std::next(ArrayQualIt),
                                          ArrayQualEnd });

      ++ArrayInfoIt;
    } break;

    default:
      revng_abort();
    }
  }

  Result = ScoredIndices{
    .Score = DifferenceScore{ .Difference = RestOff.getSExtValue(),
                              .Depth = ResultIndices.size(),
                              .InRange = RestOff.isNonNegative() },
    .Indices = std::move(ResultIndices),
  };

  return Result;
}

static std::pair<TypedAccessPattern, ChildIndexVector>
pickBestTAP(const model::QualifiedType &BaseType,
            const IRAccessPattern &IRPattern,
            const TAPToChildIdsMap &TAPIndices,
            model::VerifyHelper &VH) {
  revng_log(ModelGEPLog, "Picking Best TAP for IRAP: " << IRPattern);
  auto Indent = LoggerIndent{ ModelGEPLog };
  revng_log(ModelGEPLog, "TAPIndices.size() = " << TAPIndices.size());

  std::map<TypedAccessPattern, ChildIndexVector> BestTAPsWithIndices;

  DifferenceScore BestDifferenceScore;
  revng_log(ModelGEPLog, "BestDifferenceScore = " << BestDifferenceScore);
  auto MoreIndent = LoggerIndent{ ModelGEPLog };

  for (const auto &TAPWithIndices : TAPIndices) {

    if (ModelGEPLog.isEnabled()) {
      revng_log(ModelGEPLog, "TAP = " << TAPWithIndices.first);
      revng_log(ModelGEPLog, "Indices = {");
      for (const auto &I : TAPWithIndices.second) {
        auto InternalIndent = LoggerIndent{ ModelGEPLog };
        revng_log(ModelGEPLog, I);
      }
      revng_log(ModelGEPLog, "}");
    }
    auto EvenMoreIndent = LoggerIndent{ ModelGEPLog };

    ScoredIndices ScoredIdx = differenceScore(BaseType,
                                              TAPWithIndices,
                                              IRPattern,
                                              VH);
    if (not ScoredIdx.Score.has_value()) {
      revng_log(ModelGEPLog, "differenceScore = std::nullopt");
      continue;
    }

    DifferenceScore Difference = ScoredIdx.Score.value();

    revng_log(ModelGEPLog, "differenceScore = " << Difference);
    if (Difference > BestDifferenceScore) {
      revng_log(ModelGEPLog, "Worse than BestDifferenceScore");
      continue;
    }

    if (Difference < BestDifferenceScore) {
      BestTAPsWithIndices.clear();
      BestDifferenceScore = Difference;
      revng_log(ModelGEPLog,
                "Update BestDifferenceScore = " << BestDifferenceScore);
    }

    revng_log(ModelGEPLog, "NEW Best Indices");

    BestTAPsWithIndices[TAPWithIndices.first] = std::move(ScoredIdx.Indices);
  }

  // TODO: here we always pick the first among those with the best similarity
  // score. In the future we can try to figure out if there is a better policy.
  // But in principle we should be able to integrate the policy into the
  // similarity score, more than adding another layer of decision making here.
  revng_assert(not BestTAPsWithIndices.empty());
  return *BestTAPsWithIndices.begin();
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
getType(ModelGEPArgs &GEPArgs, model::VerifyHelper &VH) {
  std::optional<model::QualifiedType> CurrType = std::nullopt;

  if (GEPArgs.RestOff.ugt(0))
    return CurrType;

  CurrType = GEPArgs.BaseAddress.Type;
  for (const auto &[Index, AggregateType] : GEPArgs.IndexVector) {

    switch (AggregateType) {

    case AggregateKind::Struct: {

      CurrType = peelConstAndTypedefs(CurrType.value(), VH);
      auto *S = cast<model::StructType>(CurrType->UnqualifiedType.get());
      size_t FieldOffset = cast<ConstantInt>(Index)->getZExtValue();
      CurrType = S->Fields.at(FieldOffset).Type;

    } break;

    case AggregateKind::Union: {

      CurrType = peelConstAndTypedefs(CurrType.value(), VH);
      auto *U = cast<model::UnionType>(CurrType->UnqualifiedType.get());
      size_t FieldID = cast<ConstantInt>(Index)->getZExtValue();
      CurrType = U->Fields.at(FieldID).Type;

    } break;

    case AggregateKind::Array: {

      auto It = CurrType->Qualifiers.begin();

      do {
        CurrType = peelConstAndTypedefs(CurrType.value(), VH);

        It = llvm::find_if(CurrType->Qualifiers, model::Qualifier::isArray);

        // Assert that we're not skipping any pointer qualifier.
        // That would mean that the GEPArgs.IndexVector is broken w.r.t. the
        // GEPArgs.BaseAddress.
        revng_assert(not std::any_of(CurrType->Qualifiers.begin(),
                                     It,
                                     model::Qualifier::isPointer));

      } while (It == CurrType->Qualifiers.end());

      // For arrays we don't need to look at the value of the index, we just
      // unwrap the array and go on.
      CurrType = model::QualifiedType(CurrType->UnqualifiedType,
                                      { std::next(It),
                                        CurrType->Qualifiers.end() });

    } break;

    default:
      revng_abort();
    }
  }

  return CurrType;
}

static std::optional<ModelGEPArgs>
makeBestGEPArgs(const TypedBaseAddress &TBA,
                const IRAccessPattern &IRPattern,
                const TAPToChildIdsMap &TAPIndices,
                const model::Binary &Model,
                model::VerifyHelper &VH) {
  std::optional<ModelGEPArgs> Result = std::nullopt;
  LLVMContext &Ctxt = TBA.Address->getContext();

  revng_log(ModelGEPLog, "===============================");
  revng_log(ModelGEPLog, "makeBestGEPArgs for TBA: " << TBA);
  auto MakeBestGEPArgsIndent = LoggerIndent(ModelGEPLog);

  const auto &[BestTAP,
               BestIndices] = pickBestTAP(TBA.Type, IRPattern, TAPIndices, VH);

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

  const GEPSummationVector &IRPatternIndices = IRPattern.Indices;
  auto IRIndicesIt = IRPatternIndices.begin();
  auto IRIndicesEnd = IRPatternIndices.end();

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
      // For arrays we have to fill up info about the index of the array
      // access. It can be a constant or an llvm::Value, but it should never
      // be already initialized.
      revng_assert(not Id.Index);
      revng_assert(Back.Type == AggregateKind::Array);
      revng_assert(CurrentType.isArray());

      model::QualifiedType Array = peelConstAndTypedefs(CurrentType, VH);
      auto ArrayQualIt = Array.Qualifiers.begin();
      auto QEnd = Array.Qualifiers.end();

      revng_assert(ArrayQualIt != QEnd);
      revng_assert(model::Qualifier::isArray(*ArrayQualIt));

      auto *Unqualified = Array.UnqualifiedType.get();

      model::QualifiedType
        ElementType = model::QualifiedType(Model.getTypePath(Unqualified),
                                           { std::next(ArrayQualIt), QEnd });

      // We've found the first array qualifier, for which we don't know the
      // index that is being accessed. That information is stored in the
      // IRAccessPattern indices.
      // We have to unwrap it and put data about it into Indices.

      // First of all, the rest of the offset needs to be smaller than the
      // array type size.
      revng_assert(RestOff.ule(*CurrentType.size()));

      // Second, the TAP needs to still have non-consumed info associated to
      // arrays
      revng_assert(TAPArrayIt != TAPArrayEnd);

      // The array in BestTAP that we're unwrapping has a stride equal to
      // the size of this array element.
      uint64_t ElementSize = *ElementType.size();
      revng_assert(TAPArrayIt->Stride == ElementSize);

      // The array in BestTAP that we're unwrapping has the same number of
      // elements.

      // The sizes of the array should be equal, but TAPArrayIt->NumElems
      // could be 1 less than necessary because of some workarounds we have
      // built in DLA to handle arrays.
      revng_assert(ArrayQualIt->Size == TAPArrayIt->NumElems
                   or (ArrayQualIt->Size - 1 == TAPArrayIt->NumElems));

      if (RestOff.uge(ElementSize)) {
        // If the remaining offset is larger than or equal to an element size,
        // we have to compute the exact index of the element that is being
        // accessed
        APInt ElementIndex;
        APInt::udivrem(RestOff,
                       APInt(/*bitwidth*/ 64, /*value*/ ElementSize),
                       ElementIndex,
                       RestOff);
        Back.Index = ConstantInt::get(llvm::IntegerType::get(Ctxt,
                                                             64 /*NumBits*/),
                                      ElementIndex);

      } else {
        // Here the remaining offset is smaller than an element size.
        // So we have to look for a non-constant index.

        if (IRIndicesIt != IRIndicesEnd) {
          const auto &[Coefficient, Index] = *IRIndicesIt;

          // This should never happen because of how IRAccessPattern is built
          revng_assert(not isa<ConstantInt>(Index));

          // Coefficient should always have the same value of the element, so
          // that the Index is actually the index in the array.
          revng_assert(Coefficient->getValue() == ElementSize);

          Back.Index = Index;

          // The current IR indices have been handled, increase the iterator.
          ++IRIndicesIt;
        } else {
          // If we ran out of indices in IRAP there's no offset left, so
          // consider this to be zero
          revng_assert(RestOff.isNullValue());

          Back.Index = ConstantInt::get(llvm::IntegerType::get(Ctxt,
                                                               64 /*NumBits*/),
                                        RestOff);
        }
      }

      // After we're done with an array, we update CurrentType and continue to
      // the next iteration of the for loop on BestIndices, because we could
      // have another array index and another array qualifier left in
      // CurrentType
      CurrentType = ElementType;
      revng_assert(RestOff.ule(*CurrentType.size()));

      // We also omve the TAPArrayIt to point to the next array info available
      // in BestTAP
      ++TAPArrayIt;
      continue;

    } break;

    case AggregateKind::Struct: {
      const model::StructType *Struct = nullptr;

      while (not Struct) {
        // Skip over all the qualifiers. We only expect const qualifiers here.
        // And we can basically ignore them.
        for (const auto &Q : CurrentType.Qualifiers)
          revng_assert(not model::Qualifier::isPointer(Q)
                       and not model::Qualifier::isArray(Q));

        auto *Unqualified = CurrentType.UnqualifiedType.getConst();
        Struct = dyn_cast<model::StructType>(Unqualified);
        // If this is Unqualified was not a struct, the only valid thing for
        // it is to be a Typedef, in which case we unwrap it and keep looking
        // for a struct
        if (not Struct) {
          auto *TD = cast<model::TypedefType>(Unqualified);
          CurrentType = TD->UnderlyingType;
        }
      }

      // Index represents the offset of a field in the struct
      uint64_t FieldOff = cast<ConstantInt>(Back.Index)->getZExtValue();

      // The offset of the field should be smaller or equal to the remaining
      // offset. If it's not it means that the IRAP has not sufficient offset to
      // reach the pattern described by TAP, and we have to bail out.
      if (RestOff.ult(FieldOff))
        return Result;

      APInt OffsetInField = RestOff - FieldOff;
      auto &FieldType = Struct->Fields.at(FieldOff).Type;
      if (OffsetInField.uge(*FieldType.size())) {
        Result = ModelGEPArgs{ .BaseAddress = TBA,
                               .IndexVector = std::move(Indices),
                               .RestOff = RestOff,
                               .PointeeType = CurrentType };
        return Result;
      }

      // Then we subtract the field offset from the remaining offset
      RestOff = OffsetInField;
      CurrentType = FieldType;
    } break;

    case AggregateKind::Union: {
      const model::UnionType *Union = nullptr;

      while (not Union) {
        // Skip over all the qualifiers. We only expect const qualifiers here.
        // And we can basically ignore them.
        for (const auto &Q : CurrentType.Qualifiers)
          revng_assert(not model::Qualifier::isPointer(Q)
                       and not model::Qualifier::isArray(Q));

        auto *Unqualified = CurrentType.UnqualifiedType.get();
        Union = dyn_cast<model::UnionType>(Unqualified);
        // If this is Unqualified was not a union, the only valid thing for
        // it is to be a Typedef, in which case we unwrap it and keep looking
        // for a union
        if (not Union) {
          auto *TD = cast<model::TypedefType>(Unqualified);
          CurrentType = TD->UnderlyingType;
        }
      }

      // Index represents the number of the field in the union, this does not
      // affect the RestOff, since traversing union fields does not increase
      // the offset.
      uint64_t FieldId = cast<ConstantInt>(Back.Index)->getZExtValue();
      auto &FieldType = Union->Fields.at(FieldId).Type;
      if (RestOff.uge(*FieldType.size())) {
        Result = ModelGEPArgs{ .BaseAddress = TBA,
                               .IndexVector = std::move(Indices),
                               .RestOff = RestOff,
                               .PointeeType = CurrentType };
        return Result;
      }

      CurrentType = FieldType;

    } break;

    default:
      revng_abort();
    }
  }

  revng_assert(RestOff.isNonNegative());
  Result = ModelGEPArgs{ .BaseAddress = TBA,
                         .IndexVector = std::move(Indices),
                         .RestOff = RestOff,
                         .PointeeType = CurrentType };

  return Result;
}

class GEPSummationCache {

  const model::Binary &Model;

  // This maps Uses to ModelGEPSummations so that in consecutive iterations on
  // consecutive instructions we can reuse parts of them without walking the
  // entire def-use chain.
  UseGEPSummationMap UseGEPSummations = {};

  RecursiveCoroutine<ModelGEPSummation>
  getGEPSumImpl(Use &AddressUse, const ValueModelTypesMap &PointerTypes) {
    revng_log(ModelGEPLog,
              "getGEPSumImpl for use of: " << dumpToString(AddressUse.get()));
    LoggerIndent Indent{ ModelGEPLog };

    ModelGEPSummation Result = {};

    // If it's already been handled, we already know if it can be modelGEPified
    // or not, so we stick to that decision.
    auto GEPItHint = UseGEPSummations.lower_bound(&AddressUse);
    if (GEPItHint != UseGEPSummations.end()
        and not(&AddressUse < GEPItHint->first)) {
      revng_log(ModelGEPLog, "Found!");
      rc_return GEPItHint->second;
    }
    revng_log(ModelGEPLog, "Not found. Compute one!");

    Value *AddressArith = AddressUse.get();
    // If the used value and we know it has a pointer type, we already know both
    // the base address and the pointer type.
    if (auto TypeIt = PointerTypes.find(AddressArith);
        TypeIt != PointerTypes.end()) {

      revng_log(ModelGEPLog, "Use is typed!");

      auto &[AddressVal, Type] = *TypeIt;

      revng_assert(Type.isPointer());

      Result = ModelGEPSummation{
        .BaseAddress = TypedBaseAddress{ .Type = dropPointerRecursively(Type),
                                         .Address = AddressArith },
        // The summation is empty since AddressArith has exactly the type
        // we're looking at here.
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
        // In any case, we might want to devise smarter policies to discriminate
        // between different base addresses.
        // Anyway it's not clear if we can ever do something better than this.
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

        if ((nullptr != Op0Const) xor (nullptr != Op1Const)) {
          auto *ConstOp = Op1Const ? Op1Const : Op0Const;
          auto *OtherOp = Op1Const ? Op0Const : Op1Const;

          // The constant operand is the coefficient, while the other is the
          // index.
          Result = ModelGEPSummation{
            // The base address is unknown
            .BaseAddress = TypedBaseAddress{ .Type = {}, .Address = nullptr },
            // The summation has only one element, with a coefficient of 1,
            // and the index is the current instructions.
            .Summation = { ModelGEPSummationElement{ .Coefficient = ConstOp,
                                                     .Index = OtherOp } }
          };
        } else {
          // In all the other cases, fall back to treating this as a non-address
          // and non-strided instruction, just like e.g. division.

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

            // The first operand of the shift is the index
            auto *IndexForStridedAccess = AddrArithmeticInst->getOperand(0);

            Result = ModelGEPSummation{
              // The base address is unknown
              .BaseAddress = TypedBaseAddress{ .Type = {}, .Address = nullptr },
              // The summation has only one element, with a coefficient of 1,
              // and the index is the current instructions.
              .Summation = { ModelGEPSummationElement{
                .Coefficient = Stride, .Index = IndexForStridedAccess } }
            };
            // Then we're done, break from the switch
            break;
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

      case Instruction::Trunc:
      case Instruction::Load:
      case Instruction::Call:
      case Instruction::PHI:
      case Instruction::Select:
      case Instruction::ExtractValue:
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
        // address, but it's just considered as regular offset arithmetic of an
        // unknown offset.
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

      // If we reach this point the constant int does not represent a pointer so
      // we initialize the result as if it was an offset
      if (Const->getValue().isNonNegative())
        Result = makeOffsetGEPSummation(Const);

    } else if (auto *Arg = dyn_cast<Argument>(AddressArith)) {

      // If we reach this point the argument does not represent a pointer so
      // we initialize the result as if it was an offset
      Result = makeOffsetGEPSummation(Arg);

    } else if (isa<GlobalVariable>(AddressArith)) {

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
    size_t PointerBytes = getPointerSize(Model.Architecture);
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
  getGEPSummation(Use &AddressUse, const ValueModelTypesMap &PointerTypes) {
    return getGEPSumImpl(AddressUse, PointerTypes);
  }
};

class TypedAccessCache {
  QualifiedTypeToTAPChildIdsMap TAPCache;

  // Builds a map of all the possible TypedAccessPattern starting from
  // BaseType, mapping them to the vector of child indices that need to be
  // traversed on the type system to access types represented by that TAP.
  RecursiveCoroutine<TAPToChildIdsMapConstRef>
  getTAPImpl(const model::QualifiedType &BaseType, LLVMContext &Ctxt) {

    revng_log(ModelGEPLog,
              "getTAPImpl for BaseType: " << serializeToString(BaseType));
    auto Indent = LoggerIndent{ ModelGEPLog };

    auto It = TAPCache.lower_bound(BaseType);
    // If we cannot find it we have to build it
    if (It == TAPCache.end() or TAPCache.key_comp()(BaseType, It->first)) {
      revng_log(ModelGEPLog, "Not found. Build it!");
      auto MoreIndent = LoggerIndent{ ModelGEPLog };

      // Initialize a new map, that we need to fill with the results for
      // BaseType.
      TAPToChildIdsMap Result;

      // First, we need to build a new TAP representing the access pattern to
      // BaseType itself
      TypedAccessPattern NewTAP = {
        // The BaseOffset is 0, since this TAP represents an access to the
        // entire BaseType starting from BaseType itself.
        .BaseOffset = APInt(/*NumBits*/ 64, /*Value*/ 0),
        // We have no arrays info, since this TAP represents an access to the
        // entire BaseType starting from BaseType itself.
        .Arrays = {},
        // The pointee is just the BaseType
        .AccessedType = std::move(BaseType),
      };

      // The new TAP has no associated child ids, since its not accessing any
      // child of the BaseType, but the type itself
      Result[NewTAP] = {};

      if (BaseType.Qualifiers.empty()) {
        revng_log(ModelGEPLog, "No qualifiers!");
        auto EvenMoreIndent = LoggerIndent{ ModelGEPLog };
        const model::Type *BaseT = BaseType.UnqualifiedType.get();

        switch (BaseT->Kind) {

          // If we've reached a primitive type or an enum type we're done. The
          // NewTAP added above to Results is enough and we don't need to
          // traverse anything.
        case model::TypeKind::Primitive: {
          revng_log(ModelGEPLog, "Primitive. Done!");
        } break;
        case model::TypeKind::Enum: {
          revng_log(ModelGEPLog, "Enum. Done!");
        } break;

        case model::TypeKind::Struct: {
          revng_log(ModelGEPLog, "Struct, look at fields");
          const auto *S = cast<model::StructType>(BaseT);
          auto StructIndent = LoggerIndent{ ModelGEPLog };
          for (const model::StructField &Field : S->Fields) {
            revng_log(ModelGEPLog, "Field at offset: " << Field.Offset);
            auto FieldIndent = LoggerIndent{ ModelGEPLog };

            // First, traverse each child's type to get the TAPs from it
            TAPToChildIdsMap FieldResult = rc_recur getTAPImpl(Field.Type,
                                                               Ctxt);

            revng_log(ModelGEPLog,
                      "Number of types inside field: " << FieldResult.size());
            // Then, create a ChildInfo representing the traversal of the
            // children. In particular, this has a known index, that
            // represents the offset of the field in the struct.
            ChildInfo CI{
              .Index = ConstantInt::get(llvm::IntegerType::get(Ctxt,
                                                               64 /*NumBits*/),
                                        Field.Offset /*Value*/),
              .Type = AggregateKind::Struct
            };

            // Now iterate on data in InnerResult, and massage them to add the
            // field offset to the TAP, as well as the child info to the child
            // ids, before actually merging them into Result.
            auto InnerTAPIt = FieldResult.begin();
            auto InnerTAPEnd = FieldResult.end();
            while (InnerTAPIt != InnerTAPEnd) {
              // Save the next valid value of the iterator, because we're
              // going to extract the pointee of InnerTAPIt and mess with it
              // before inserting into Result, and that would make it
              // impossible to properly continue the iteration on InnerResult
              // otherwise.
              auto InnerTAPNext = std::next(InnerTAPIt);

              auto TAPWithIdsHandle = FieldResult.extract(InnerTAPIt);

              // Add the Field.Offset to the Base offset
              auto &BaseOffset = TAPWithIdsHandle.key().BaseOffset;
              BaseOffset += Field.Offset;

              // Prepend the info on this struct to the child ids in the inner
              // result.
              auto &ChildIds = TAPWithIdsHandle.mapped();
              ChildIds.insert(ChildIds.begin(), CI);

              Result.insert(std::move(TAPWithIdsHandle));

              // Increment the iterator.
              InnerTAPIt = InnerTAPNext;
            }
          }
        } break;

        case model::TypeKind::Union: {
          revng_log(ModelGEPLog, "Union, look at fields");
          const auto *U = cast<model::UnionType>(BaseT);
          auto UnionIndent = LoggerIndent{ ModelGEPLog };
          for (const model::UnionField &Field : U->Fields) {
            revng_log(ModelGEPLog, "Field ID: " << Field.Index);
            auto FieldIndent = LoggerIndent{ ModelGEPLog };

            // First, traverse each child's type to get the TAPs from it
            TAPToChildIdsMap FieldResult = rc_recur getTAPImpl(Field.Type,
                                                               Ctxt);

            revng_log(ModelGEPLog,
                      "Number of types inside field: " << FieldResult.size());
            // Then, create a ChildInfo representing the traversal of the
            // children. In particular, this has a known index, that
            // represents the number of the field in the struct (not its
            // offset in this case)
            ChildInfo CI{
              .Index = ConstantInt::get(llvm::IntegerType::get(Ctxt,
                                                               64 /*NumBits*/),
                                        Field.Index /*Value*/),
              .Type = AggregateKind::Union
            };

            // Now iterate on data in InnerResult, and massage them to add the
            // field offset to the TAP, as well as the child info to the child
            // ids, before actually merging them into Result.
            auto InnerTAPIt = FieldResult.begin();
            auto InnerTAPEnd = FieldResult.end();
            while (InnerTAPIt != InnerTAPEnd) {
              // Save the next valid value of the iterator, because we're
              // going to extract the pointee of InnerTAPIt and mess with it
              // before inserting into Result, and that would make it
              // impossible to properly continue the iteration on InnerResult
              // otherwise.
              auto InnerTAPNext = std::next(InnerTAPIt);

              auto TAPWithIdsHandle = FieldResult.extract(InnerTAPIt);

              // Prepend the info on this struct to the child ids in the inner
              // result.
              auto &ChildIds = TAPWithIdsHandle.mapped();
              ChildIds.insert(ChildIds.begin(), CI);

              Result.insert(std::move(TAPWithIdsHandle));

              // Increment the iterator.
              InnerTAPIt = InnerTAPNext;
            }
          }
        } break;

        case model::TypeKind::Typedef: {
          revng_log(ModelGEPLog, "Typedef, unwrap");
          // For typedefs, we need to unwrap the underlying type and try to
          // traverse it.
          const auto *TD = cast<model::TypedefType>(BaseT);
          TAPToChildIdsMap InnerResult = rc_recur getTAPImpl(TD->UnderlyingType,
                                                             Ctxt);
          // The InnerResult can just be merged into the Result, because
          // typedefs are shallow names that don't really add ids to the
          // traversal of the typesystem.
          Result.merge(std::move(InnerResult));
        } break;

        case model::TypeKind::RawFunctionType:
        case model::TypeKind::CABIFunctionType: {
          revng_abort();
        } break;

        default:
          revng_abort();
        }
      } else {
        revng_log(ModelGEPLog,
                  "Has qualifiers: " << BaseType.Qualifiers.size());
        auto EvenMoreIndent = LoggerIndent{ ModelGEPLog };
        const model::Qualifier &FirstQualifier = *BaseType.Qualifiers.begin();

        // If the first qualifier is a pointer qualifier, we're done
        // descending, because the pointee does not reside into the BaseType,
        // it's only referenced by it. In all the other cases (arrays and
        // const) we need to unwrap the first layer (qualifier) and keep
        // looking for other TAPs that might be generated.
        if (not model::Qualifier::isPointer(FirstQualifier)) {
          revng_log(ModelGEPLog, "FirstQualifier is not ConstQualifier");

          auto QIt = std::next(BaseType.Qualifiers.begin());
          auto QEnd = BaseType.Qualifiers.end();
          auto InnerType = model::QualifiedType(BaseType.UnqualifiedType,
                                                { QIt, QEnd });

          // First, compute the InnerResult, which represents all the TAPs
          // from the InnerType going downward. At this point we do make a
          // copy of it, because we'll need to change it with information on
          // BaseType
          TAPToChildIdsMap InnerResult = rc_recur getTAPImpl(InnerType, Ctxt);

          if (not model::Qualifier::isConst(FirstQualifier)) {
            // If the first qualifier is const, we can just use the
            // InnerResult for BaseType as well.
            // Otherwise, the first qualifier is an array, and we need to
            // handle that.
            revng_assert(model::Qualifier::isArray(FirstQualifier));
            revng_log(ModelGEPLog, "FirstQualifier is not ConstQualifier");

            // First, build the array info associated to the array we're
            // handling.
            uint64_t NElems = FirstQualifier.Size;
            revng_assert(InnerType.size());
            uint64_t Stride = *InnerType.size();
            ArrayInfo AI{ .Stride = APInt(/*NumBits*/ 64, /*Value*/ Stride),
                          .NumElems = APInt(/*NumBits*/ 64,
                                            /*Value*/ NElems) };

            // Second, build the child info associated to the array we're
            // handling. In this case we initialize the Index to nullptr,
            // because at this point we don't really know the index used for
            // accessing the array. This will be fixed up later, whenever we
            // have elected the best TypedAccessPattern or the given
            // IRAccessPattern. At that point the Index will be expanded with
            // an actual llvm::Value.
            ChildInfo CI{ .Index = nullptr, .Type = AggregateKind::Array };

            // Now iterate on data in InnerResult, and massage them to add
            // array info before actually merging them into Result.
            auto InnerTAPIt = InnerResult.begin();
            auto InnerTAPEnd = InnerResult.end();
            while (InnerTAPIt != InnerTAPEnd) {
              // Save the next valid value of the iterator, because we're
              // going to extract the pointee of InnerTAPIt and mess with it
              // before inserting into Result, and that would make it
              // impossible to properly continue the iteration on InnerResult
              // otherwise.
              auto InnerTAPNext = std::next(InnerTAPIt);

              auto TAPWithIdsHandle = InnerResult.extract(InnerTAPIt);

              // Prepend the info on this array to the Arrays info in the
              // inner result.
              auto &Arrays = TAPWithIdsHandle.key().Arrays;
              Arrays.insert(Arrays.begin(), AI);

              // Prepend the info on this array to the child ids in the inner
              // result.
              auto &ChildIds = TAPWithIdsHandle.mapped();
              ChildIds.insert(ChildIds.begin(), CI);

              Result.insert(std::move(TAPWithIdsHandle));

              // Increment the iterator.
              InnerTAPIt = InnerTAPNext;
            }
          }
        }
      }
      revng_log(ModelGEPLog, "Result.size() = " << Result.size());
      It = TAPCache.insert(It, { BaseType, std::move(Result) });
    } else {
      revng_log(ModelGEPLog, "Found!");
    }

    rc_return It->second;
  }

public:
  const TAPToChildIdsMap &
  getTAP(const model::QualifiedType &BaseType, LLVMContext &Ctxt) {
    return static_cast<TAPToChildIdsMapConstRef>(getTAPImpl(BaseType, Ctxt))
      .get();
  }

  void clear() { TAPCache.clear(); }
};

using UseGEPInfoMap = std::map<Use *, ModelGEPArgs>;

static UseGEPInfoMap
makeGEPReplacements(llvm::Function &F, const model::Binary &Model) {

  UseGEPInfoMap Result;

  const model::Function *ModelF = llvmToModelFunction(Model, F);
  revng_assert(ModelF);

  // First, try to initialize a map for the known model types of llvm::Values
  // that are reachable from F. If this fails, we just bail out because we
  // cannot infer any modelGEP in F, if we have no type information to rely on.
  ValueModelTypesMap PointerTypes = initializeModelTypes(F, *ModelF, Model);
  if (PointerTypes.empty()) {
    revng_log(ModelGEPLog, "Model Types not found for " << F.getName());
    return Result;
  }

  GEPSummationCache GEPSumCache{ Model };
  TypedAccessCache TAPCache;
  model::VerifyHelper VH;

  LLVMContext &Ctxt = F.getContext();
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

        // Skip callee operands in CallInst
        if (auto *CallUser = dyn_cast<CallInst>(U.getUser())) {
          if (&U == &CallUser->getCalledOperandUse()) {
            revng_log(ModelGEPLog, "Skipping callee operand in CallInst");
            continue;
          }
        }

        // Skip all but the pointer operands of load and store instructions
        if (auto *Load = dyn_cast<LoadInst>(U.getUser())) {
          if (U.getOperandNo() != Load->getPointerOperandIndex()) {
            revng_log(ModelGEPLog, "Skipping non-pointer operand in LoadInst");
            continue;
          }
        }

        if (auto *Store = dyn_cast<StoreInst>(U.getUser())) {
          if (U.getOperandNo() != Store->getPointerOperandIndex()) {
            revng_log(ModelGEPLog, "Skipping non-pointer operand in LoadInst");
            continue;
          }
        }

        // Skip booleans, since they cannot be addresses
        if (auto *IntTy = dyn_cast<llvm::IntegerType>(U.get()->getType())) {
          if (IntTy->getIntegerBitWidth() == 1) {
            revng_log(ModelGEPLog, "Skipping i1 value");
            continue;
          }
        }

        ModelGEPSummation GEPSum = GEPSumCache.getGEPSummation(U, PointerTypes);

        revng_log(ModelGEPLog, "GEPSum " << GEPSum);
        if (not GEPSum.isAddress())
          continue;

        // Pre-compute all the typed access patterns from the base address of
        // the GEPSum, or get them from the caches if we've already computed
        // them.
        const model::QualifiedType &BaseTy = GEPSum.BaseAddress.Type;
        const auto &TAPToChildIds = TAPCache.getTAP(BaseTy, Ctxt);

        // If the set of typed access patterns from BaseTy is empty we can skip
        // to the next instruction
        if (TAPToChildIds.empty())
          continue;

        // Now we extract an IRAccessPattern from the ModelGEPSummation
        IRAccessPattern IRPattern = computeAccessPattern(U, GEPSum, Model);

        // Select among the computed TAPIndices the one which best fits the
        // IRPattern
        auto BestGEPArgsOrNone = makeBestGEPArgs(GEPSum.BaseAddress,
                                                 IRPattern,
                                                 TAPToChildIds,
                                                 Model,
                                                 VH);

        // If the selection failed, we bail out.
        if (not BestGEPArgsOrNone.has_value())
          continue;

        ModelGEPArgs &GEPArgs = BestGEPArgsOrNone.value();

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
          std::optional<model::QualifiedType> GEPTypeOrNone = getType(GEPArgs,
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

  std::map<model::QualifiedType, Constant *, QTLess> GlobalModelGEPTypeArgs;

public:
  Value *getQualifiedTypeArg(model::QualifiedType &QT, llvm::Module &M) {
    auto It = GlobalModelGEPTypeArgs.find(QT);
    if (It != GlobalModelGEPTypeArgs.end())
      return It->second;

    std::string SerializedQT;
    {
      llvm::raw_string_ostream StringStream(SerializedQT);
      llvm::yaml::Output YAMLOutput(StringStream);
      YAMLOutput << QT;
    }
    It = GlobalModelGEPTypeArgs
           .insert({ QT, buildStringPtr(&M, SerializedQT, "") })
           .first;
    return It->second;
  }
};

bool MakeModelGEPPass::runOnFunction(llvm::Function &F) {
  bool Changed = false;

  // Skip non-isolated functions
  if (not FunctionTags::Isolated.isTagOf(&F))
    return Changed;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.hasName() or not F.getName().equals(TargetFunction.c_str()))
      return Changed;

  revng_log(ModelGEPLog, "Make ModelGEP for " << F.getName());
  auto Indent = LoggerIndent(ModelGEPLog);

  auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();

  UseGEPInfoMap GEPReplacementMap = makeGEPReplacements(F, *Model);

  llvm::Module &M = *F.getParent();
  LLVMContext &Ctxt = M.getContext();
  IRBuilder<> Builder(Ctxt);
  ModelGEPArgCache TypeArgCache;
  for (auto &[TheUseToGEPify, GEPArgs] : GEPReplacementMap) {

    revng_log(ModelGEPLog,
              "GEPify use of: " << dumpToString(TheUseToGEPify->get()));
    revng_log(ModelGEPLog,
              "  `-> use in: " << dumpToString(TheUseToGEPify->getUser()));

    llvm::Type *UseType = TheUseToGEPify->get()->getType();
    llvm::Type *BaseAddrType = GEPArgs.BaseAddress.Address->getType();
    auto *ModelGEPFunction = getModelGEP(M, UseType, BaseAddrType);

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

    // The other arguments are the indices in IndexVector
    for (auto [ChildId, AggregateTy] : GEPArgs.IndexVector) {
      revng_assert(isa<ConstantInt>(ChildId)
                   or AggregateTy == AggregateKind::Array);
      Args.push_back(ChildId);
    }

    // Insert a call to ModelGEP right before the use, special casing the
    // uses that are incoming for PHI nodes.
    auto *UserInstr = cast<Instruction>(TheUseToGEPify->getUser());
    if (auto *PHIUser = dyn_cast<PHINode>(UserInstr)) {
      auto *IncomingB = PHIUser->getIncomingBlock(*TheUseToGEPify);
      Builder.SetInsertPoint(IncomingB->getTerminator());
    } else {
      Builder.SetInsertPoint(UserInstr);
    }

    Value *ModelGEPRef = Builder.CreateCall(ModelGEPFunction, Args);

    auto *AddressOfFunction = getAddressOf(M, UseType);
    auto *PointeeConstantStrPtr = TypeArgCache
                                    .getQualifiedTypeArg(GEPArgs.PointeeType,
                                                         M);
    Value *ModelGEPPtr = Builder.CreateCall(AddressOfFunction,
                                            { PointeeConstantStrPtr,
                                              ModelGEPRef });

    if (GEPArgs.RestOff.isStrictlyPositive()) {
      // If the GEPArgs have a RestOff that is strictly positive, we have to
      // inject the remaining part of the pointer arithmetic as normal sums
      revng_assert(UseType->isIntOrPtrTy());

      // First, cast it to int if necessary.
      if (UseType->isPointerTy()) {
        auto *IntType = llvm::IntegerType::get(Ctxt,
                                               GEPArgs.RestOff.getBitWidth());
        ModelGEPPtr = Builder.CreatePtrToInt(ModelGEPPtr, IntType);
      }

      // Then, inject the actuall add
      auto GEPResultBitWidth = ModelGEPPtr->getType()->getIntegerBitWidth();
      APInt OffsetToAdd = GEPArgs.RestOff.zextOrTrunc(GEPResultBitWidth);
      ModelGEPPtr = Builder.CreateAdd(ModelGEPPtr,
                                      ConstantInt::get(Ctxt, OffsetToAdd));

      // Finally, convert it back to pointer.
      if (UseType->isPointerTy())
        ModelGEPPtr = Builder.CreateIntToPtr(ModelGEPPtr, UseType);
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
