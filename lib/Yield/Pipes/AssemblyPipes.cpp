//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Doxygen.h"
#include "revng/PTML/Tag.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldAssembly.h"

namespace revng::pypeline::piperuns {

ProcessAssembly::ProcessAssembly(const class Model &Model,
                                 llvm::StringRef Config,
                                 llvm::StringRef DynamicConfig,
                                 const BinariesContainer &BinariesContainer,
                                 const CFGMap &CFG,
                                 AssemblyInternalContainer &Output) :
  Binary(*Model.get().get()), CFG(CFG), Output(Output), NameBuilder(Binary) {
  Helper = std::make_unique<DissassemblyHelper>();

  auto BinaryBuffer = BinariesContainer.getFile(0);
  BinaryView = std::make_unique<RawBinaryView>(Binary,
                                               llvm::StringRef{
                                                 BinaryBuffer.data(),
                                                 BinaryBuffer.size() });
};

ProcessAssembly::~ProcessAssembly() = default;

void ProcessAssembly::runOnFunction(const model::Function &TheFunction) {
  ObjectID Object(TheFunction.Entry());
  const auto &Metadata = CFG.getElement(Object);

  TupleTree<yield::Function> &OutputFunction = Output.getElement(Object);
  Helper->disassemble(TheFunction,
                      *Metadata,
                      *BinaryView,
                      Binary,
                      NameBuilder,
                      *OutputFunction);
}

void YieldAssembly::runOnFunction(const model::Function &TheFunction) {
  MetaAddress Address = TheFunction.Entry();
  ObjectID Object(Address);
  const TupleTree<yield::Function> &Function = Input.getElement(Object);

  revng_assert(Function.verify());
  revng_assert(Function->verify());
  revng_assert(Function->Entry() == Address);

  const model::Architecture::Values A = Model.Architecture();
  auto CommentIndicator = model::Architecture::getAssemblyCommentIndicator(A);

  const model::Configuration &Configuration = Model.Configuration();
  uint64_t LineWidth = Configuration.CommentLineWidth();

  std::string R = ptml::functionComment(B,
                                        TheFunction,
                                        Model,
                                        CommentIndicator,
                                        0,
                                        LineWidth,
                                        NameBuilder);
  R += yield::ptml::functionAssembly(B, *Function, Model);
  R = B.getTag(ptml::tags::Div, std::move(R)).toString();
  *Output.getOStream(Object) << R;
}

} // namespace revng::pypeline::piperuns
