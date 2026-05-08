//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/PTML/CAttributes.h"
#include "revng/PTML/CBuilder.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/TypeNames/LLVMTypeNames.h"

using llvm::dyn_cast;

static Logger Log{ "helpers-to-header" };

/// Print the declaration a C struct corresponding to an LLVM struct
/// type.
static void printLLVMTypeDeclaration(const llvm::StructType *S,
                                     const llvm::Function &F,
                                     ptml::ModelCBuilder &B) {
  B.append(B.getKeyword(ptml::CBuilder::Keyword::Typedef) + " "
           + B.getKeyword(ptml::CBuilder::Keyword::Struct) + " "
           + ptml::AttributeRegistry::getAttributeString<"_PACKED">() + " ");

  {
    auto Scope = B.getCurvedBracketScope(ptml::c::scopes::StructBody.str());

    const auto &DataLayout = F.getParent()->getDataLayout();
    const auto *StructLayout = DataLayout.getStructLayout(S);
    uint64_t StructByteSize = StructLayout->getSizeInBytes();

    uint64_t PreviousOffsetInBits = 0ULL;

    for (auto [Index, FieldType] : llvm::enumerate(S->elements())) {
      revng_assert(PreviousOffsetInBits % 8 == 0);

      uint64_t FieldOffsetInBits = StructLayout->getElementOffsetInBits(Index);
      revng_assert(FieldOffsetInBits % 8 == 0);

      B.printPadding(PreviousOffsetInBits / 8, FieldOffsetInBits / 8);

      B.append(getReturnStructFieldTypeReferenceTag(&F, Index, B) + " "
               + getReturnStructFieldDefinitionTag(&F, Index, B) + ";\n");

      uint64_t FieldSizeInBits = DataLayout.getTypeSizeInBits(FieldType);
      revng_assert(FieldSizeInBits % 8 == 0,
                   (F.getName().str()
                    + " returns a StructType with a field whose size "
                      "is not a multiple of 8 bits. Field index: "
                    + std::to_string(Index)
                    + ". FieldSizeInBits: " + std::to_string(FieldSizeInBits))
                     .c_str());

      PreviousOffsetInBits = FieldOffsetInBits + FieldSizeInBits;
    }
    revng_assert(PreviousOffsetInBits % 8 == 0);
    revng_assert(PreviousOffsetInBits / 8 == StructByteSize);
  }

  B.append(" " + getReturnTypeDefinitionTag(&F, B) + ";\n");
}

/// Print the prototype of a helper.
static void printHelperPrototype(const llvm::Function *Func,
                                 ptml::ModelCBuilder &B) {
  B.append(getReturnTypeReferenceTag(Func, B) + " "
           + getHelperFunctionDefinitionTag(Func, B));

  if (Func->arg_empty()) {
    B.append("(" + B.getVoidTag() + ");\n");

  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &Arg : Func->args()) {
      B.append(Separator.str() + getScalarTypeTag(Arg.getType(), B));
      Separator = Comma;
    }
    B.append(");\n");
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

bool ptml::HeaderBuilder::printHelpersHeader(const llvm::Module &M) {
  auto Scope = B.getScopeTag(ptml::tags::Div);
  std::string Includes = B.getPragmaOnce() + B.getIncludeAngle("stdint.h")
                         + B.getIncludeAngle("stdbool.h")
                         + B.getIncludeQuote("primitive-types.h") + "\n";
  B.append(std::move(Includes));

  std::set<uint64_t> OpaqueByteSizes;

  for (const llvm::Function &F : M.functions()) {

    // Skip non-helpers
    bool IsHelper = FunctionTags::QEMU.isTagOf(&F)
                    or FunctionTags::Helper.isTagOf(&F)
                    or FunctionTags::OpaqueCSVValue.isTagOf(&F)
                    or FunctionTags::Exceptional.isTagOf(&F) or F.isIntrinsic();
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
    // TODO: we should try to drop this check
    if (hasUnprintableArgsOrRetTypes(F))
      continue;

    // TODO: this is a wrong usage of Log. Convert to cl::opt or drop.
    if (Log.isEnabled()) {
      auto CommentScope = B.getLineCommentScope();
      std::string Serialized{};
      llvm::raw_string_ostream Helper{ Serialized };
      Helper << *F.getType();
      Helper.flush();
      B.append(std::move(Serialized));
    }

    // Print the declaration of the return type, if it's not scalar
    const auto *RetTy = F.getReturnType();
    if (auto *RetStructTy = dyn_cast<llvm::StructType>(RetTy)) {
      printLLVMTypeDeclaration(RetStructTy, F, B);

      const auto &DataLayout = F.getParent()->getDataLayout();
      const auto *StructLayout = DataLayout.getStructLayout(RetStructTy);
      uint64_t ByteSize = StructLayout->getSizeInBytes();
      revng_assert(StructLayout->getSizeInBits() == 8 * ByteSize);
      OpaqueByteSizes.insert(StructLayout->getSizeInBytes());

      B.append("\n");
    }

    for (auto &Arg : F.args()) {
      revng_assert(Arg.getType()->isSingleValueType());
    }

    printHelperPrototype(&F, B);

    B.append("\n");
  }

  // Always print the opaque type definitions here.
  // The helper header is only used if we have the function body, which will
  // need these definitions.
  B.appendLineComment("\\defgroup Opaque Types");
  B.appendLineComment("\\{");

  auto OpaqueTypeScope = B.getScopeTag(ptml::tags::Div);

  B.printHelperOpaqueTypeDefinitions(OpaqueByteSizes);

  B.appendLineComment("\\}");
  B.append("\n");

  return true;
}
