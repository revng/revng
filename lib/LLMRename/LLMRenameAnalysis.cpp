//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Process.h"

#include "revng/LLMRename/LLMRenameAnalysis.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ProgramRunner.h"
#include "revng/TupleTree/VisitsImpl.h"

static llvm::Error doRename(model::Binary &Model, llvm::StringRef Contents) {
  using ModelT = model::Binary;
  ProgramRunner::RunOptions Options{
    .Stdin = Contents,
    .Capture = ProgramRunner::CaptureOption::StdoutAndStderrSeparately
  };
  ProgramRunner::Result Result = ::Runner.run("revng",
                                              { "llm-rename" },
                                              Options);
  if (Result.ExitCode == 2) {
    // Exit code 2 is treated specially to propagate stderr as the error
    // message
    return revng::createError(Result.Stderr);
  } else if (Result.ExitCode != 0) {
    return revng::createError("Failed to run llm-rename, process "
                              "returned with exit code: "
                              + std::to_string(Result.ExitCode));
  }

  auto MaybeChanges = fromString<TupleTreeDiff<ModelT>>(Result.Stdout);
  if (not MaybeChanges)
    return MaybeChanges.takeError();

  for (const Change<ModelT> &Change : MaybeChanges->Changes) {
    bool SetResult = setByPath(Change.Path, Model, *Change.New);
    if (not SetResult)
      return revng::createError("Could not apply change");
  }

  return llvm::Error::success();
}

namespace revng::pypeline::analyses {

llvm::Error LLMRename::run(Model &Model,
                           const Request &Incoming,
                           llvm::StringRef Configuration,
                           const PTMLCFunctionContainer &Input) {
  model::Binary &Binary = *Model.get().get();
  for (const ObjectID *Object : Incoming[0]) {
    auto Buffer = Input.getMemoryBuffer(*Object);
    if (auto Error = doRename(Binary, Buffer->getBuffer()))
      return Error;
  }
  return llvm::Error::success();
}

bool LLMRename::isAvailable() const {
  auto Env = llvm::sys::Process::GetEnv("OPENAI_API_KEY");
  return Env.has_value() and not Env->empty();
}

} // namespace revng::pypeline::analyses
