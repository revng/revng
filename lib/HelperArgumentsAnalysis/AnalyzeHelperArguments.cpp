/// \file AnalyzeHelperArguments.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "AnnotationWriter.h"
#include "ArgumentUsageAnalysis.h"
#include "CPUStateUsage.h"
#include "Context.h"
#include "Function.h"
#include "Value.h"
#include "llvm-c/Types.h"

using namespace llvm;

static cl::opt<std::string> ModuleDumpPath("analyze-helper-arguments-output",
                                           cl::desc("path where to save the "
                                                    "module annotated with the "
                                                    "results of the Analyze "
                                                    "Helper Arguments Analysis"
                                                    "results"),
                                           cl::init(""));

// TODO: use ConstantExpr to perform the constant folding instead of inventing
//       our own arithmetic framework

static aua::StructPointers collectOffsetTypes(Module &Module,
                                              StructType &ArchCPU) {
  LLVMContext &Context = Module.getContext();
  aua::StructPointers Result(Module, ArchCPU);

  for (Function &F : Module) {
    for (Instruction &I : llvm::instructions(F)) {
      auto *Dbg = dyn_cast<DbgValueInst>(&I);
      if (Dbg == nullptr)
        continue;

      auto *MAV0 = dyn_cast<MetadataAsValue>(Dbg->getArgOperand(0));
      auto *MAV1 = cast<MetadataAsValue>(Dbg->getArgOperand(1))->getMetadata();
      if (MAV0 == nullptr or MAV1 == nullptr
          or not isa<ValueAsMetadata>(MAV0->getMetadata())
          or not isa<DILocalVariable>(MAV1))
        continue;

      auto *Value = cast<ValueAsMetadata>(MAV0->getMetadata())->getValue();
      // TODO: go beyond arguments?
      auto *Argument = dyn_cast_or_null<llvm::Argument>(Value);
      auto *Type = cast<DILocalVariable>(MAV1)->getType();
      if (Argument == nullptr or Type == nullptr)
        continue;

      revng_log(Log,
                "Considering argument " << Argument->getArgNo() << " of "
                                        << F.getName().str());
      LoggerIndent Indent(Log);

      auto *DIArgumentType = dyn_cast_or_null<DIDerivedType>(Type);
      if (DIArgumentType == nullptr
          or DIArgumentType->getTag() != dwarf::DW_TAG_pointer_type) {
        revng_log(Log, "Not a pointer");
        continue;
      }

      auto *BaseType = DIArgumentType->getBaseType();

      // Skip typedefs
      auto *DerivedType = dyn_cast_or_null<DIDerivedType>(BaseType);
      while (DerivedType != nullptr
             and DerivedType->getTag() == llvm::dwarf::DW_TAG_typedef) {
        BaseType = DerivedType->getBaseType();
        DerivedType = dyn_cast_or_null<DIDerivedType>(BaseType);
      }

      if (BaseType == nullptr or BaseType->getRawName() == nullptr) {
        revng_log(Log, "Pointee has no name");
        continue;
      }

      for (const char *Prefix : std::vector{ "struct.", "union." }) {
        auto *Struct = StructType::getTypeByName(Context,
                                                 (Prefix + BaseType->getName())
                                                   .str());

        if (Struct != nullptr) {
          revng_log(Log, "Registering as a pointer to " << Struct->getName());
          Result.registerPointer(*Value, *Struct);
        }
      }
    }
  }

  Result.propagateFromActualArguments();

  if (Log.isEnabled()) {
    Result.dump(Log);
    Log << DoLog;
  }

  return Result;
}

class AnalyzeHelperArguments : public ModulePass {
public:
  static char ID;

public:
  AnalyzeHelperArguments() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    Task T(6, "Analyze helper arguments");

    T.advance("Prepare module");
    LLVMContext &LLVMContext = M.getContext();
    auto &DL = M.getDataLayout();

    // NOTE: patch these things on the QEMU side wherever possible!
    std::array NoOpFunctionNames = {
      // Ignore debugging primitives
      "printf",

      // Ignore assertion-like
      "g_assertion_message_expr",
    };
    std::vector<Function *> NoOpFunctions;
    for (auto &&Name : NoOpFunctionNames)
      if (Function *F = M.getFunction(Name))
        NoOpFunctions.push_back(F);

    // NOTE: patch these things on the QEMU side wherever possible!
    std::array AbortFunctionNames = {
      // glib features we don't support
      "g_hash_table_lookup",
      "g_string_free",
    };
    std::vector<Function *> AbortFunctions;
    for (auto &&Name : AbortFunctionNames)
      if (Function *F = M.getFunction(Name))
        AbortFunctions.push_back(F);

    for (Function &F : M) {
      if (F.getSection() == "revng_noop") {
        NoOpFunctions.push_back(&F);
      } else if (F.getSection() == "revng_abort") {
        AbortFunctions.push_back(&F);
      }
    }

    for (Function *F : NoOpFunctions) {
      auto *ReturnType = F->getFunctionType()->getReturnType();
      revng_assert(ReturnType->isVoidTy() or ReturnType->isPointerTy()
                   or ReturnType->isIntegerTy());

      // Purge the function body
      {
        for (BasicBlock &BB : *F)
          BB.dropAllReferences();
        while (not F->empty())
          F->begin()->eraseFromParent();
      }

      auto *Block = BasicBlock::Create(LLVMContext, "", F);
      if (ReturnType->isVoidTy()) {
        ReturnInst::Create(LLVMContext, Block);
      } else if (ReturnType->isPointerTy()) {
        auto *Null = ConstantPointerNull::get(cast<PointerType>(ReturnType));
        ReturnInst::Create(LLVMContext, Null, Block);
      } else {
        auto *Zero = ConstantInt::get(ReturnType, 0);
        ReturnInst::Create(LLVMContext, Zero, Block);
      }
    }

    for (Function *F : AbortFunctions) {
      {
        for (BasicBlock &BB : *F)
          BB.dropAllReferences();
        while (not F->empty())
          F->begin()->eraseFromParent();
      }

      F->setDoesNotReturn();
      auto *Block = BasicBlock::Create(LLVMContext, "", F);
      new UnreachableInst(LLVMContext, Block);
    }

    // Remove noreturn from cpu_loop_exit and its call sites
    for (auto &F : M) {
      if (F.getName().starts_with("cpu_loop_exit")) {
        F.removeFnAttr(Attribute::NoReturn);

        for (CallBase *Call : callers(&F))
          Call->removeFnAttr(Attribute::NoReturn);
      }
    }

    aua::Context Context;

    T.advance("Run argument usage analysis");
    aua::ArgumentUsageAnalysis AUA(Context, M);
    AUA.run();

    T.advance("Collect offsets types");
    auto *CPUStruct = StructType::getTypeByName(LLVMContext, "struct.ArchCPU");
    aua::StructPointers OffsetTypes = collectOffsetTypes(M, *CPUStruct);

    T.advance("Analyzing helpers");
    aua::CPUStateUsageAnalysis CSUA(Context,
                                    AUA,
                                    DL,
                                    *CPUStruct,
                                    std::move(OffsetTypes));

    SmallVector<Function *> HelperDefinitions;
    for (Function &F : M)
      if (not F.isDeclaration() and F.getName().starts_with("helper_"))
        HelperDefinitions.push_back(&F);

    {
      Task T(HelperDefinitions.size(), "Analyzing helpers");
      for (Function *F : HelperDefinitions) {
        T.advance(F->getName());
        revng_assert(not F->isVarArg());
        CSUA.analyze(*F);
      }
    }

    T.advance("Adding annotations");
    CSUA.annotate(M);

    T.advance("Dumping module");
    if (ModuleDumpPath.getNumOccurrences() != 0) {
      aua::AnnotationWriter Annotator = aua::AnnotationWriter(AUA, CSUA);
      std::error_code EC;
      raw_fd_ostream Stream(ModuleDumpPath.getValue(), EC);
      revng_assert(not EC);
      M.print(Stream, &Annotator);
    }

    return false;
  }
};

char AnalyzeHelperArguments::ID = 0;

using Register = RegisterPass<AnalyzeHelperArguments>;
static Register X("analyze-helper-arguments",
                  "Analyze usage of arguments of helper functions");
