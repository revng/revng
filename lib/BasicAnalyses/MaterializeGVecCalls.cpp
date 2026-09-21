//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <optional>
#include <vector>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Support/IRBuilder.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;

namespace {

using RegisterMap = std::map<uint64_t, GlobalVariable *>;

static std::optional<uint64_t> cpuStateOffset(Value *Pointer) {
  if (Pointer->getType()->getPointerAddressSpace() != 0)
    return std::nullopt;
  if (isa<ConstantPointerNull>(Pointer))
    return 0;
  auto *Address = dyn_cast<ConstantExpr>(Pointer);
  if (Address == nullptr or Address->getOpcode() != Instruction::IntToPtr)
    return std::nullopt;
  auto *Offset = dyn_cast<ConstantInt>(Address->getOperand(0));
  if (Offset == nullptr or Offset->getBitWidth() > 64)
    return std::nullopt;
  return Offset->getZExtValue();
}

/// Find the CSV containing a CPU-state byte. Offsets come from VariableManager,
/// not from register names or a target-specific CPU layout.
static std::pair<GlobalVariable *, uint64_t>
locate(const RegisterMap &Registers, uint64_t Offset) {
  auto It = Registers.upper_bound(Offset);
  if (It == Registers.begin())
    return { nullptr, 0 };
  --It;
  uint64_t Byte = Offset - It->first;
  if (Byte >= It->second->getValueType()->getIntegerBitWidth() / 8)
    return { nullptr, 0 };
  return { It->second, Byte };
}

/// Reject references to helper runtime state, including references hidden in
/// constant expressions. Only the CSV declarations may cross modules.
static bool isExportable(Value *V, const ValueToValueMapTy &ExportMap) {
  if (auto *GV = dyn_cast<GlobalValue>(V))
    return ExportMap.count(GV);
  if (auto *C = dyn_cast<Constant>(V))
    for (Value *Operand : C->operands())
      if (not isExportable(Operand, ExportMap))
        return false;
  return true;
}

/// Specialize in a separate module: optimizing root's CFG would discard its
/// markers, and temporary allocations violate the outliner's contract.
static Function *specialize(CallInst &Original,
                            Module &Helpers,
                            const RegisterMap &Registers,
                            ValueToValueMapTy &ExportMap,
                            StringRef &Reason) {
  auto Fail = [&](StringRef Message) -> Function * {
    Reason = Message;
    return nullptr;
  };
  auto *Helper = Helpers.getFunction(Original.getCalledFunction()->getName());
  if (Helper == nullptr
      or Helper->getFunctionType() != Original.getFunctionType()
      or not Helper->getReturnType()->isVoidTy())
    return Fail("missing helper body or unsupported signature");

  SmallVector<Type *> Types = { PointerType::getUnqual(Helpers.getContext()) };
  for (Value *Argument : Original.args()) {
    if (Argument->getType()->isPointerTy()) {
      auto Offset = cpuStateOffset(Argument);
      if (not Offset)
        return Fail("pointer is not a constant CPU-state offset");
      if (locate(Registers, *Offset).first == nullptr)
        return Fail("pointer has no CPU-state variable");
    } else if (not isa<Constant>(Argument)) {
      Types.push_back(Argument->getType());
    }
  }

  auto *FT = FunctionType::get(Type::getVoidTy(Helpers.getContext()),
                               Types,
                               false);
  auto *Wrapper = Function::Create(FT,
                                   GlobalValue::InternalLinkage,
                                   "materialized_gvec",
                                   Helpers);
  auto Cleanup = make_scope_exit([&]() { Wrapper->eraseFromParent(); });
  auto *Memory = Wrapper->getArg(0);
  auto *Block = BasicBlock::Create(Helpers.getContext(), "entry", Wrapper);
  revng::IRBuilder Builder(Block);
  SmallVector<Value *> Arguments;
  unsigned Next = 1;
  for (Value *Argument : Original.args()) {
    if (Argument->getType()->isPointerTy()) {
      auto *Offset = Builder.getInt64(*cpuStateOffset(Argument));
      // An unknown allocation preserves aliases without imposing an artificial
      // object size. Validate its actual accesses after specialization.
      Arguments.push_back(Builder.CreateGEP(Builder.getInt8Ty(),
                                            Memory,
                                            Offset));
    } else if (isa<Constant>(Argument)) {
      Arguments.push_back(Argument);
    } else {
      Arguments.push_back(Wrapper->getArg(Next++));
    }
  }
  Builder.CreateCall(Helper, Arguments);
  Builder.CreateRetVoid();

  // Limit inlining and unrolling for unexpectedly large helper dependencies.
  constexpr unsigned MaxInstructions = 16384;
  for (unsigned Round = 0; true; ++Round) {
    SmallVector<CallInst *> Inline;
    unsigned Size = Wrapper->getInstructionCount();
    for (Instruction &I : instructions(Wrapper)) {
      auto *Call = dyn_cast<CallInst>(&I);
      auto *F = Call == nullptr ? nullptr : Call->getCalledFunction();
      if (F == nullptr or F->isIntrinsic())
        continue;
      if (auto Error = F->materialize()) {
        consumeError(std::move(Error));
        return Fail("cannot materialize helper body");
      }
      if (F->isDeclaration())
        return Fail("helper depends on an external function");
      Size += F->getInstructionCount();
      Inline.push_back(Call);
    }
    if (Inline.empty())
      break;
    if (Round == 16 or Size > MaxInstructions)
      return Fail("helper exceeds the specialization budget");
    for (CallInst *Call : Inline) {
      InlineFunctionInfo Info;
      if (not InlineFunction(*Call, Info, false, nullptr, false).isSuccess())
        return Fail("cannot inline helper body");
    }
  }

  legacy::FunctionPassManager PM(&Helpers);
  PM.add(createSROAPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createLoopSimplifyPass());
  PM.add(createLCSSAPass());
  PM.add(createLoopUnrollPass(3,
                              false,
                              false,
                              MaxInstructions,
                              -1,
                              0,
                              0,
                              1,
                              0));
  PM.add(createSROAPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.doInitialization();
  PM.run(*Wrapper);
  PM.doFinalization();
  stripDebugInfo(*Wrapper);

  // First validate every access. A failure must leave the lifted call intact.
  SmallVector<std::pair<Instruction *, uint64_t>> Accesses;
  for (Instruction &I : instructions(Wrapper)) {
    Value *Pointer = nullptr;
    Type *Type = nullptr;
    if (auto *Load = dyn_cast<LoadInst>(&I)) {
      if (not Load->isSimple())
        return Fail("volatile or atomic helper access");
      Pointer = Load->getPointerOperand();
      Type = Load->getType();
    } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
      if (not Store->isSimple())
        return Fail("volatile or atomic helper access");
      Pointer = Store->getPointerOperand();
      Type = Store->getValueOperand()->getType();
    } else if (isa<AllocaInst>(I) or isa<CallBase>(I)) {
      return Fail("helper did not fully specialize");
    } else {
      continue;
    }
    auto *Integer = dyn_cast<IntegerType>(Type);
    int64_t Offset = 0;
    auto *Base = GetPointerBaseWithConstantOffset(Pointer,
                                                  Offset,
                                                  Helpers.getDataLayout());
    if (Integer == nullptr or Integer->getBitWidth() % 8 != 0 or Base != Memory
        or Offset < 0)
      return Fail("helper access is not a constant integer CPU-state access");
    for (unsigned Byte = 0; Byte < Integer->getBitWidth() / 8; ++Byte)
      if (locate(Registers, Offset + Byte).first == nullptr)
        return Fail("helper access crosses unmapped CPU-state bytes");
    Accesses.emplace_back(&I, Offset);
  }

  // Preserve the helper's access order, including read/modify/write operations.
  for (auto [I, Offset] : Accesses) {
    Builder.SetInsertPoint(I);
    auto *Load = dyn_cast<LoadInst>(I);
    auto *Store = dyn_cast<StoreInst>(I);
    auto *Type = cast<IntegerType>(Load != nullptr ?
                                     Load->getType() :
                                     Store->getValueOperand()->getType());
    Value *Result = ConstantInt::get(Type, 0);
    for (unsigned Byte = 0; Byte < Type->getBitWidth() / 8; ++Byte) {
      auto [CSV, Part] = locate(Registers, Offset + Byte);
      auto *CSVType = cast<IntegerType>(CSV->getValueType());
      auto *Old = Builder.CreateLoad(CSVType, CSV);
      if (Load != nullptr) {
        auto *Shifted = Builder.CreateLShr(Old, Part * 8);
        auto *Truncated = Builder.CreateTrunc(Shifted, Builder.getInt8Ty());
        auto *Extended = Builder.CreateZExtOrTrunc(Truncated, Type);
        Result = Builder.CreateOr(Result,
                                  Builder.CreateShl(Extended, Byte * 8));
      } else {
        auto *Value = Builder.CreateLShr(Store->getValueOperand(), Byte * 8);
        auto *Truncated = Builder.CreateTrunc(Value, Builder.getInt8Ty());
        auto *Extended = Builder.CreateZExtOrTrunc(Truncated, CSVType);
        auto *Shifted = Builder.CreateShl(Extended, Part * 8);
        APInt Mask = APInt::getBitsSet(CSVType->getBitWidth(),
                                       Part * 8,
                                       Part * 8 + 8);
        auto *Kept = Builder.CreateAnd(Old, ConstantInt::get(CSVType, ~Mask));
        Builder.CreateStore(Builder.CreateOr(Kept, Shifted), CSV);
      }
    }
    if (Load != nullptr)
      Load->replaceAllUsesWith(Result);
    I->eraseFromParent();
  }

  legacy::FunctionPassManager CleanupPM(&Helpers);
  // Forward partial writes before removing overwritten stores. This avoids
  // introducing a false dependency on a fully overwritten destination CSV.
  CleanupPM.add(createEarlyCSEPass(true));
  CleanupPM.add(createInstructionCombiningPass());
  CleanupPM.add(createDeadStoreEliminationPass());
  CleanupPM.add(createCFGSimplificationPass());
  CleanupPM.doInitialization();
  CleanupPM.run(*Wrapper);
  CleanupPM.doFinalization();
  if (not Memory->use_empty())
    return Fail("helper retains a CPU-state pointer");
  for (Instruction &I : instructions(Wrapper))
    for (Value *Operand : I.operands())
      if (not isExportable(Operand, ExportMap))
        return Fail("helper retains a reference to runtime state");

  auto *ScalarFT = FunctionType::get(FT->getReturnType(),
                                     ArrayRef(Types).drop_front(),
                                     false);
  auto *Scalarized = Function::Create(ScalarFT,
                                      GlobalValue::InternalLinkage,
                                      "scalarized_gvec",
                                      *Original.getModule());
  ValueToValueMapTy Map;
  for (auto Entry : ExportMap)
    Map[Entry.first] = Entry.second;
  Map[Memory] = PoisonValue::get(Memory->getType());
  for (unsigned I = 1; I < Wrapper->arg_size(); ++I)
    Map[Wrapper->getArg(I)] = Scalarized->getArg(I - 1);
  SmallVector<ReturnInst *> Returns;
  CloneFunctionInto(Scalarized,
                    Wrapper,
                    Map,
                    CloneFunctionChangeType::DifferentModule,
                    Returns);
  // Cloning can create an empty compile-unit list even after stripping debug
  // information. Do not export it to a module without debug-version flags.
  auto *CU = Original.getModule()->getNamedMetadata("llvm.dbg.cu");
  if (CU != nullptr and CU->getNumOperands() == 0)
    CU->eraseFromParent();
  return Scalarized;
}

/// QEMU's generic vector helpers take void pointers, so the typed CPU-pointer
/// analysis cannot identify their register effects. Expose them before ABI
/// analysis discards apparently unused register definitions.
class MaterializeGVecCalls : public ModulePass {
public:
  static char ID;
  MaterializeGVecCalls() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    SmallVector<CallInst *> Calls;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (auto *Callee = Call->getCalledFunction())
            if (Callee->getName().starts_with("helper_gvec_"))
              Calls.push_back(Call);
    if (Calls.empty())
      return false;

    auto Skip = [](StringRef Reason) {
      errs() << "warning: materialize-gvec-calls: " << Reason
             << "; retaining generic vector helper calls\n";
      return false;
    };
    if (M.getDataLayout().isBigEndian())
      return Skip("big-endian CPU-state storage is unsupported");

    std::unique_ptr<Module> Helpers;
    if (llvm::any_of(Calls, [](CallInst *C) {
          return C->getCalledFunction()->isDeclaration();
        })) {
      auto *Arch = M.getNamedMetadata("revng.qemu_architecture");
      if (Arch == nullptr or Arch->getNumOperands() == 0
          or Arch->getOperand(0)->getNumOperands() != 1)
        return Skip("missing helper architecture");
      auto *Name = dyn_cast<MDString>(Arch->getOperand(0)->getOperand(0));
      if (Name == nullptr)
        return Skip("invalid helper architecture");
      for (MDNode *Node : Arch->operands())
        if (Node->getNumOperands() != 1 or Node->getOperand(0) != Name)
          return Skip("conflicting helper architectures");
      auto File = revng::ResourceFinder.findFile(("/share/revng/"
                                                  "libtcg-helpers-full-"
                                                  + Name->getString() + ".bc")
                                                   .str());
      if (not File)
        return Skip("cannot find helper bitcode");
      SMDiagnostic Diagnostic;
      auto Library = getLazyIRFileModule(*File, Diagnostic, M.getContext());
      if (not Library)
        return Skip("cannot read helper bitcode");
      // Import only referenced helpers and their dependencies. The full library
      // is large, and optimization requires a fully materialized module.
      Helpers = std::make_unique<Module>("gvec-helpers", M.getContext());
      Helpers->setDataLayout(Library->getDataLayout());
      Helpers->setTargetTriple(Library->getTargetTriple());
      auto *FT = FunctionType::get(Type::getVoidTy(M.getContext()), false);
      auto *References = Function::Create(FT,
                                          GlobalValue::InternalLinkage,
                                          "gvec_references",
                                          *Helpers);
      revng::IRBuilder Builder(BasicBlock::Create(M.getContext(),
                                                  "",
                                                  References));
      for (CallInst *Call : Calls) {
        Function *F = Call->getCalledFunction();
        if (Helpers->getFunction(F->getName()) != nullptr)
          continue;
        auto *Declaration = Function::Create(F->getFunctionType(),
                                             GlobalValue::ExternalLinkage,
                                             F->getName(),
                                             *Helpers);
        SmallVector<Value *> Arguments;
        for (Type *Type : F->getFunctionType()->params())
          Arguments.push_back(PoisonValue::get(Type));
        Builder.CreateCall(Declaration, Arguments);
      }
      Builder.CreateRetVoid();
      if (Linker::linkModules(*Helpers,
                              std::move(Library),
                              Linker::Flags::LinkOnlyNeeded))
        return Skip("cannot import helper bodies");
      References->eraseFromParent();
    } else {
      Helpers = CloneModule(M);
    }
    if (Helpers->getDataLayout().isBigEndian())
      return Skip("big-endian helper storage is unsupported");

    RegisterMap Registers;
    ValueToValueMapTy ExportMap;
    for (GlobalVariable &GV : M.globals()) {
      auto *MD = GV.getMetadata("revng.csv.offset");
      if (MD == nullptr)
        continue;
      auto *Type = dyn_cast<IntegerType>(GV.getValueType());
      auto *Offset = MD->getNumOperands() == 1 ?
                       mdconst::dyn_extract<ConstantInt>(MD->getOperand(0)) :
                       nullptr;
      if (Type == nullptr or Type->getBitWidth() % 8 != 0 or Offset == nullptr
          or Offset->getBitWidth() > 64 or GV.isConstant()
          or GV.getAddressSpace() != 0
          or Offset->getZExtValue() > INT64_MAX - Type->getBitWidth() / 8)
        return Skip("invalid CPU-state offset metadata");
      auto *Copy = new GlobalVariable(*Helpers,
                                      Type,
                                      false,
                                      GlobalValue::ExternalLinkage,
                                      nullptr,
                                      "materialized_csv");
      if (not Registers.emplace(Offset->getZExtValue(), Copy).second)
        return Skip("duplicate CPU-state offsets");
      ExportMap[Copy] = &GV;
    }
    uint64_t End = 0;
    for (auto [Offset, CSV] : Registers) {
      if (Offset < End)
        return Skip("overlapping CPU-state variables");
      End = Offset + CSV->getValueType()->getIntegerBitWidth() / 8;
    }

    // Repeated instructions commonly use the same registers and descriptor.
    // Specialize once per helper/constant-argument tuple, including failures.
    std::map<std::vector<Constant *>, Function *> Cache;
    bool Changed = false;
    for (CallInst *Original : Calls) {
      std::vector<Constant *> Key = { Original->getCalledFunction() };
      SmallVector<Value *> Arguments;
      for (Value *Argument : Original->args()) {
        Key.push_back(dyn_cast<Constant>(Argument));
        if (not isa<Constant>(Argument))
          Arguments.push_back(Argument);
      }
      auto [It, New] = Cache.emplace(std::move(Key), nullptr);
      if (New) {
        StringRef Reason;
        It->second = specialize(*Original,
                                *Helpers,
                                Registers,
                                ExportMap,
                                Reason);
        if (It->second == nullptr)
          errs() << "warning: materialize-gvec-calls: retaining "
                 << Original->getCalledFunction()->getName() << ": " << Reason
                 << "\n";
      }
      if (It->second == nullptr)
        continue;
      revng::IRBuilder Replacement(Original);
      auto *Call = Replacement.CreateCall(It->second, Arguments);
      InlineFunctionInfo Info;
      if (not InlineFunction(*Call, Info, false, nullptr, false).isSuccess()) {
        Call->eraseFromParent();
        continue;
      }
      Original->eraseFromParent();
      Changed = true;
    }
    for (auto &[Key, Function] : Cache)
      if (Function != nullptr)
        Function->eraseFromParent();
    return Changed;
  }
};

char MaterializeGVecCalls::ID = 0;
RegisterPass<MaterializeGVecCalls> X("materialize-gvec-calls",
                                     "Expose generic vector helper CPU-state "
                                     "effects",
                                     false,
                                     false);

} // namespace
