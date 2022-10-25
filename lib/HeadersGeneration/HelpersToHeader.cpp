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
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

static Logger<> Log{ "helpers-to-header" };

/// Print the declaration a C struct corresponding to an LLVM struct
/// type.
static void printDefinition(const llvm::StructType *S,
                            const VariableTokensWithName Struct,
                            ptml::PTMLIndentedOstream &Header) {
  Header << keywords::Typedef << " " << keywords::Struct << " "
         << helpers::Packed << " ";
  {
    Scope Scope(Header, scopeTags::Struct);

    for (const auto &Field : llvm::enumerate(S->elements())) {
      auto Info = getFieldInfo(S, Field.index());
      Header
        << ptml::tokenTag(Info.FieldTypeName, tokens::Type) << " "
        << Tag(tags::Span, Info.FieldName.str())
             .addAttribute(attributes::Token, tokens::Field)
             .addAttribute(attributes::LocationDefinition,
                           serializedLocation(ranks::HelperStructField,
                                              Struct.VariableName.str().str(),
                                              Info.FieldName.str().str()))
        << ";\n";
    }
  }

  Header << " " << Struct.Declaration << ";\n";
}

/// Print the prototype of a helper .
static void printHelperPrototype(const llvm::Function *Func,
                                 ptml::PTMLIndentedOstream &Header) {
  Header << getReturnType(Func).Use << " "
         << ptml::tokenTag(model::Identifier::fromString(Func->getName()).str(),
                           tokens::Function)
              .addAttribute(attributes::LocationDefinition,
                            serializedLocation(ranks::Helpers,
                                               Func->getName().str()));

  if (Func->arg_empty()) {
    Header << "(" + ptml::tokenTag("void", tokens::Type) + ");\n";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &Arg : Func->args()) {
      auto Type = getScalarCType(Arg.getType());
      std::string TypeTag;
      if (Type.endswith(" *")) {
        Type.pop_back_n(2);
        TypeTag = ptml::tokenTag(Type.str(), tokens::Type) + " "
                  + operators::PointerDereference;
      } else {
        TypeTag = ptml::tokenTag(Type.str(), tokens::Type).serialize();
      }
      Header << Separator << TypeTag;
      Separator = Comma;
    }
    Header << ");\n";
  }
}

bool dumpHelpersToHeader(const llvm::Module &M, llvm::raw_ostream &Out) {
  auto Header = ptml::PTMLIndentedOstream(Out, 4);
  {
    auto Scope = Tag(tags::Div).scope(Header);
    Header << helpers::pragmaOnce();
    Header << helpers::includeAngle("stdint.h");
    Header << helpers::includeAngle("stdbool.h");
    Header << helpers::includeQuote("revngfloat.h");
    Header << "\n";

    for (const llvm::Function &F : M.functions()) {
      const auto &FuncName = F.getName();
      if (FunctionTags::QEMU.isTagOf(&F) or FunctionTags::Helper.isTagOf(&F)
          or F.isIntrinsic() or FuncName.startswith("llvm.")
          or FuncName.startswith("init_")
          or FuncName.startswith("revng_init_")) {

        if (Log.isEnabled()) {
          auto LineCommentScope = helpers::LineComment(Header);
          Header << *F.getType();
        }

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
  }

  return true;
}
