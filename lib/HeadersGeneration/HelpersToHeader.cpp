//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Identifier.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

using llvm::dyn_cast;

static Logger<> Log{ "helpers-to-header" };

/// Print the declaration a C struct corresponding to an LLVM struct
/// type.
static void printDefinition(const llvm::StructType *S,
                            const llvm::StringRef StructName,
                            llvm::raw_ostream &Header) {
  Header << "typedef struct __attribute__((packed)) {\n";

  for (const auto &Field : llvm::enumerate(S->elements()))
    Header << getScalarCType(Field.value()) << " "
           << getFieldName(S, Field.index()).FieldName << ";\n";

  Header << "} " << StructName << ";\n";
}

/// Print the prototype of a helper .
static void
printHelperPrototype(const llvm::Function *Func, llvm::raw_ostream &Header) {
  Header << getReturnType(Func) << " "
         << model::Identifier::fromString(Func->getName());

  if (Func->arg_empty()) {
    Header << "(void);\n";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &Arg : Func->args()) {
      Header << Separator << getScalarCType(Arg.getType());
      Separator = Comma;
    }
    Header << ");\n";
  }
}

bool dumpHelpersToHeader(const llvm::Module &M, llvm::raw_ostream &Header) {
  Header << "#pragma once\n";
  Header << "#include <stdint.h>\n";
  Header << "#include <stdbool.h>\n";
  Header << "#include \"revngfloat.h\"\n\n";

  for (const llvm::Function &F : M.functions()) {
    const auto &FuncName = F.getName();
    if (FunctionTags::QEMU.isTagOf(&F) or FunctionTags::Helper.isTagOf(&F)
        or F.isIntrinsic() or FuncName.startswith("llvm.")
        or FuncName.startswith("init_") or FuncName.startswith("revng_init_")) {

      if (Log.isEnabled())
        Header << "// " << *F.getType() << "\n";

      // Print the declaration of the return type, if it's not scalar
      const auto *RetTy = F.getReturnType();
      if (auto *RetStructTy = dyn_cast<llvm::StructType>(RetTy)) {
        const auto &StructName = getReturnType(&F);
        printDefinition(RetStructTy, StructName, Header);
      }

      for (auto &Arg : F.args())
        revng_assert(Arg.getType()->isSingleValueType());

      printHelperPrototype(&F, Header);
    }
  }

  return true;
}
