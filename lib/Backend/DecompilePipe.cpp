//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompileFunction.h"
#include "revng/Backend/DecompilePipe.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Support/Identifier.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

static llvm::cl::opt<std::string> ProblemNameFile("naming-collisions-report",
                                                  llvm::cl::desc("The file to "
                                                                 "list problem "
                                                                 "names in"),
                                                  llvm::cl::value_desc("file"
                                                                       "name"));

namespace revng::pipes {

using namespace pipeline;
static RegisterDefaultConstructibleContainer<DecompileStringMap> Reg;

static void reportProblemNames(const model::Binary &Binary) {
  if (ProblemNameFile.getValue().empty())
    return;

  std::error_code ErrorCode;
  llvm::raw_fd_ostream OutputStream(ProblemNameFile, ErrorCode);
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  auto Output = [&OutputStream](auto &&Result) {
    if (Result)
      OutputStream << *Result << '\n';
  };

  model::CNameBuilder Builder(Binary);
  for (const model::Function &F : Binary.Functions())
    Output(Builder.warning(F));
  for (const model::Segment &S : Binary.Segments())
    Output(Builder.warning(Binary, S));
  for (const model::UpcastableTypeDefinition &D : Binary.TypeDefinitions()) {
    Output(Builder.warning(*D));

    namespace M = model;
    if (auto *F = llvm::dyn_cast<M::CABIFunctionDefinition>(D.get())) {
      for (const model::Argument &A : F->Arguments())
        Output(Builder.warning(*F, A));

    } else if (auto *F = llvm::dyn_cast<M::RawFunctionDefinition>(D.get())) {
      for (const model::NamedTypedRegister &R : F->Arguments())
        Output(Builder.warning(*F, R));
      for (const model::NamedTypedRegister &R : F->ReturnValues())
        Output(Builder.warning(*F, R));

    } else if (auto *E = llvm::dyn_cast<M::EnumDefinition>(D.get())) {
      for (const model::EnumEntry &Entry : E->Entries())
        Output(Builder.warning(*E, Entry));

    } else if (auto *T = llvm::dyn_cast<M::TypedefDefinition>(D.get())) {
      // No extra names inside typedefs

    } else if (auto *S = llvm::dyn_cast<M::StructDefinition>(D.get())) {
      for (const model::StructField &Field : S->Fields())
        Output(Builder.warning(*S, Field));

    } else if (auto *U = llvm::dyn_cast<M::UnionDefinition>(D.get())) {
      for (const model::UnionField &Field : U->Fields())
        Output(Builder.warning(*U, Field));

    } else {
      revng_abort("Unsupported type definition kind.");
    }
  }

  OutputStream.flush();
  ErrorCode = OutputStream.error();
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());
}

// Does a sanity check on the names we are possibly emitting.
static std::optional<std::string>
gatherNonReservedHelperNames(const llvm::Module &Module,
                             const model::CNameBuilder &B) {
  std::string ErrorMessage;
  for (const llvm::Function &Function : Module.functions()) {
    std::string SanitizedName = sanitizeIdentifier(Function.getName());

    if (llvm::Error Error = B.isNameReserved(SanitizedName)) {
      // If we get an error here, then the identifier is reserved, which is
      // exactly what we want for helpers!
      llvm::consumeError(std::move(Error));
      continue;
    }

    if (FunctionTags::QEMU.isTagOf(&Function))
      ErrorMessage += "- QEMU: " + SanitizedName + "\n";

    if (FunctionTags::Helper.isTagOf(&Function)
        and not FunctionTags::BinaryOperationOverflows.isTagOf(&Function)) {
      ErrorMessage += "- Helper: " + SanitizedName + "\n";
    }

    if (FunctionTags::OpaqueCSVValue.isTagOf(&Function))
      ErrorMessage += "- OpaqueCSVValue: " + SanitizedName + "\n";

    if (FunctionTags::Exceptional.isTagOf(&Function))
      ErrorMessage += "- Exceptional: " + SanitizedName + "\n";

    if (Function.isIntrinsic())
      ErrorMessage += "- Intrinsic: " + SanitizedName + "\n";
  }

  if (ErrorMessage.empty())
    return std::nullopt;

  return std::move(ErrorMessage);
}

void Decompile::run(pipeline::ExecutionContext &EC,
                    pipeline::LLVMContainer &IRContainer,
                    const revng::pipes::CFGMap &CFGMap,
                    DecompileStringMap &DecompiledFunctions) {

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(EC);
  ControlFlowGraphCache Cache(CFGMap);

  namespace options = revng::options;
  ptml::CTypeBuilder
    B(llvm::nulls(),
      Model,
      /* EnableTaglessMode = */ false,
      { .EnableStackFrameInlining = options::EnableStackFrameInlining });

  for (const model::Function &Function :
       getFunctionsAndCommit(EC, DecompiledFunctions.name())) {
    auto *F = Module.getFunction(B.NameBuilder.llvmName(Function));
    std::string CCode = decompile(Cache, *F, Model, B);
    DecompiledFunctions.insert_or_assign(Function.Entry(), std::move(CCode));
  }

  reportProblemNames(Model);

  if (std::optional Names = gatherNonReservedHelperNames(Module, B.NameBuilder))
    revng_abort(("Function names should be forbidden:\n" + *Names).c_str());
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::Decompile> Y;
