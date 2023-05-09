//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Identifier.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/HeadersGeneration/HelpersToHeader.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

using llvm::dyn_cast;

using pipeline::serializedLocation;
using ptml::Tag;
namespace tags = ptml::tags;
namespace attributes = ptml::attributes;
namespace ranks = revng::ranks;

static Logger<> Log{ "helpers-to-header" };

/// Print the declaration a C struct corresponding to an LLVM struct
/// type.
static void printDefinition(const llvm::StructType *S,
                            const llvm::Function &F,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &ThePTMLCBuilder) {
  Header << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef)
         << " "
         << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Struct)
         << " " << ThePTMLCBuilder.getAttributePacked() << " ";

  {
    Scope Scope(Header, ptml::c::scopes::StructBody);

    for (const auto &Field : llvm::enumerate(S->elements())) {
      Header << getReturnStructFieldType(&F, Field.index(), ThePTMLCBuilder)
             << " "
             << getReturnStructFieldLocationDefinition(&F,
                                                       Field.index(),
                                                       ThePTMLCBuilder)
             << ";\n";
    }
  }

  Header << " " << getReturnTypeLocationDefinition(&F, ThePTMLCBuilder)
         << ";\n";
}

/// Print the prototype of a helper .
static void printHelperPrototype(const llvm::Function *Func,
                                 ptml::PTMLIndentedOstream &Header,
                                 ptml::PTMLCBuilder &ThePTMLCBuilder) {
  Header << getReturnTypeLocationReference(Func, ThePTMLCBuilder) << " "
         << getHelperFunctionLocationDefinition(Func, ThePTMLCBuilder);

  if (Func->arg_empty()) {
    Header << "(" + ThePTMLCBuilder.tokenTag("void", ptml::c::tokens::Type)
                + ");\n";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &Arg : Func->args()) {
      Header << Separator << getScalarCType(Arg.getType(), ThePTMLCBuilder);
      Separator = Comma;
    }
    Header << ");\n";
  }
}

static bool hasUnprintableArgsOrRetTypes(const llvm::Function &F) {
  const auto IsUnprintable = std::not_fn(isScalarCType);
  auto *ReturnType = F.getReturnType();
  if (not isScalarCType(ReturnType)) {
    if (auto *StructTy = dyn_cast<llvm::StructType>(ReturnType)) {
      return llvm::any_of(StructTy->elements(), IsUnprintable);
    }
    return true;
  }

  return llvm::any_of(F.getFunctionType()->params(), IsUnprintable);
}

bool dumpHelpersToHeader(const llvm::Module &M, llvm::raw_ostream &Out) {
  using PTMLCBuilder = ptml::PTMLCBuilder;
  PTMLCBuilder ThePTMLCBuilder;
  auto Header = ptml::PTMLIndentedOstream(Out, 4);
  {
    auto Scope = ThePTMLCBuilder.getTag(ptml::tags::Div).scope(Header);
    Header << ThePTMLCBuilder.getPragmaOnce();
    Header << ThePTMLCBuilder.getIncludeAngle("stdint.h");
    Header << ThePTMLCBuilder.getIncludeAngle("stdbool.h");
    Header << ThePTMLCBuilder.getIncludeQuote("revng-primitive-types.h");
    Header << "\n";

    for (const llvm::Function &F : M.functions()) {

      // Skip non-helpers
      bool IsHelper = FunctionTags::QEMU.isTagOf(&F)
                      or FunctionTags::Helper.isTagOf(&F)
                      or FunctionTags::OpaqueCSVValue.isTagOf(&F)
                      or FunctionTags::Exceptional.isTagOf(&F)
                      or F.isIntrinsic();
      if (not IsHelper)
        continue;

      // Skip helpers that should never be printed:
      // - because we expect them to never require emission and we wouldn't know
      //   how to emit them (e.g. target-specific intrinsics)
      // - because we want to actively avoid printing them even if they are
      //   present, such as all LLVM's debug intrinsics
      llvm::StringRef FName = F.getName();
      bool ShouldNotBePrinted = F.isTargetIntrinsic()
                                or FName.startswith("llvm.dbg");
      if (ShouldNotBePrinted)
        continue;

      // Skip helpers that have argument types or return types that we don't
      // know how to print (e.g. vector types, or struct types whose fields are
      // not only pointers or integers).
      // These should never happen in revng-generated IR anyway, except for some
      // leftover unused declarations of custom helpers that are never used
      // (such as unknownPC)
      if (hasUnprintableArgsOrRetTypes(F))
        continue;

      if (Log.isEnabled()) {
        auto
          LineCommentScope = helpers::LineComment(Header,
                                                  ThePTMLCBuilder
                                                    .isGenerateTagLessPTML());
        Header << *F.getType();
      }

      // Print the declaration of the return type, if it's not scalar
      const auto *RetTy = F.getReturnType();
      if (auto *RetStructTy = dyn_cast<llvm::StructType>(RetTy)) {
        printDefinition(RetStructTy, F, Header, ThePTMLCBuilder);
        Header << '\n';
      }

      for (auto &Arg : F.args()) {
        revng_assert(Arg.getType()->isSingleValueType());
      }

      printHelperPrototype(&F, Header, ThePTMLCBuilder);
      Header << '\n';
    }
  }

  return true;
}
