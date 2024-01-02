//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"

#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"

#include "revng-c/Pipes/Kinds.h"

#include "MakeSegmentRefPass.h"

namespace revng::pipes {

class MakeSegmentRefs {
public:
  static constexpr auto Name = "MakeSegmentRefs";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract BinaryPart(kinds::Binary, 0, kinds::Binary, 0);
    pipeline::Contract FunctionsPart(kinds::StackAccessesSegregated,
                                     1,
                                     kinds::StackAccessesSegregated,
                                     1);
    return { pipeline::ContractGroup({ BinaryPart, FunctionsPart }) };
  }

  void run(pipeline::ExecutionContext &Ctx,
           const BinaryFileContainer &SourceBinary,
           pipeline::LLVMContainer &Output);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const;

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    revng_check(ContainerNames.size() == 2);
    OS << *ResourceFinder.findFile("bin/revng");
    OS << " opt MakeSegmentRefs " << ContainerNames[0] << " "
       << ContainerNames[1] << "\n";
  }
};

void MakeSegmentRefs::run(pipeline::ExecutionContext &Ctx,
                          const BinaryFileContainer &SourceBinary,
                          pipeline::LLVMContainer &TargetsList) {
  if (not SourceBinary.exists())
    return;

  const TupleTree<model::Binary> &Model = getModelFromContext(Ctx);
  auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*SourceBinary.path());
  auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));
  RawBinaryView RawBinary(*Model, Buffer->getBuffer());

  llvm::legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadBinaryWrapperPass(Buffer->getBuffer()));
  PM.add(new MakeSegmentRefPass);

  PM.run(TargetsList.getModule());
}

llvm::Error
MakeSegmentRefs::checkPrecondition(const pipeline::Context &Ctx) const {
  return llvm::Error::success();
}

} // namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::MakeSegmentRefs> RegMSRPipe;
