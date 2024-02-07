/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringBufferContainer.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/YAMLTraits.h"

using namespace llvm;

struct DecoratedFunction {
  MetaAddress Entry;
  std::string OriginalName;
  efa::FunctionMetadata FunctionMetadata;
  MutableSet<model::FunctionAttribute::Values> Attributes;
};

template<>
struct llvm::yaml::MappingTraits<DecoratedFunction> {
  static void mapping(IO &TheIO, DecoratedFunction &Info) {
    TheIO.mapRequired("Entry", Info.Entry);
    TheIO.mapOptional("OriginalName", Info.OriginalName);
    TheIO.mapOptional("FunctionMetadata", Info.FunctionMetadata);
    TheIO.mapOptional("Attributes", Info.Attributes);
  }
};

template<>
struct KeyedObjectTraits<DecoratedFunction> {
  static MetaAddress key(const DecoratedFunction &Obj) { return Obj.Entry; }

  static DecoratedFunction fromKey(MetaAddress Key) {
    return DecoratedFunction{ .Entry = Key };
  }
};

namespace revng::pipes {

inline constexpr char CFGMIMEType[] = "text/x.yaml";
inline constexpr char CFGName[] = "cfg";
inline constexpr char CFGExtension[] = ".yml";
using CFGContainer = StringBufferContainer<&kinds::CFG,
                                           CFGName,
                                           CFGMIMEType,
                                           CFGExtension>;

class EmitCFG {
public:
  static constexpr auto Name = "emit-cfg";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::Isolated,
                                     0,
                                     kinds::CFG,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &Context,
           const pipeline::LLVMContainer &Module,
           CFGContainer &Output);

  Error checkPrecondition(const pipeline::Context &Ctx) const {
    return Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "(unavailable)\n";
  }
};

void EmitCFG::run(pipeline::ExecutionContext &Context,
                  const pipeline::LLVMContainer &ModuleContainer,
                  CFGContainer &Output) {

  const TupleTree<model::Binary> &Model = getModelFromContext(Context);
  const llvm::Module *Module = &ModuleContainer.getModule();

  SortedVector<DecoratedFunction> DecoratedFunctions;
  auto *RootFunction = Module->getFunction("root");
  revng_assert(RootFunction != nullptr);

  FunctionMetadataCache Cache;
  if (not RootFunction->isDeclaration()) {
    for (BasicBlock &BB : *Module->getFunction("root")) {
      llvm::Instruction *Term = BB.getTerminator();
      auto *FMMDNode = Term->getMetadata(FunctionMetadataMDName);
      if (not FMMDNode)
        continue;

      const efa::FunctionMetadata &FM = Cache.getFunctionMetadata(&BB);
      auto &Function = Model->Functions().at(FM.Entry());
      MutableSet<model::FunctionAttribute::Values> Attributes;
      for (auto &Entry : Function.Attributes())
        Attributes.insert(Entry);
      DecoratedFunction NewFunction(FM.Entry(),
                                    Function.OriginalName(),
                                    FM,
                                    Attributes);
      DecoratedFunctions.insert(std::move(NewFunction));
    }
  }

  for (const Function &F : FunctionTags::Isolated.functions(Module)) {
    auto *FMMDNode = F.getMetadata(FunctionMetadataMDName);
    const efa::FunctionMetadata &FM = Cache.getFunctionMetadata(&F);
    if (not FMMDNode or DecoratedFunctions.contains(FM.Entry()))
      continue;

    auto &Function = Model->Functions().at(FM.Entry());
    MutableSet<model::FunctionAttribute::Values> Attributes;
    for (auto &Entry : Function.Attributes())
      Attributes.insert(Entry);
    DecoratedFunction NewFunction(FM.Entry(),
                                  Function.OriginalName(),
                                  FM,
                                  Attributes);
    DecoratedFunctions.insert(std::move(NewFunction));
  }

  Output.setContent(serializeToString(DecoratedFunctions));
}

} // namespace revng::pipes

using namespace revng::pipes;

static pipeline::RegisterPipe<EmitCFG> E1;
static pipeline::RegisterDefaultConstructibleContainer<CFGContainer> Reg;
