//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Helpers/Native/Registry.h"
#include "revng/Pypeline/TraceRunner/Runner.h"
#include "revng/Pypeline/TraceRunner/Savepoint.h"

using namespace revng::pypeline::helpers::native;
using namespace revng::pypeline::tracerunner;
using ContainerMap = llvm::StringMap<std::unique_ptr<Container>>;

static std::vector<const ObjectID *>
pointerize(const std::vector<ObjectID> &Vec) {
  std::vector<const ObjectID *> Result;
  for (auto &Element : Vec)
    Result.push_back(&Element);
  return Result;
}

static llvm::ArrayRef<const ObjectID> toRef(const std::vector<ObjectID> &Vec) {
  if (Vec.size() == 0)
    return {};
  const ObjectID *Start = &*Vec.begin();
  return { Start, Vec.size() };
}

static void runTask(RegistryImpl &R,
                    ContainerMap &CMap,
                    const Model &TheModel,
                    const PipeTask &Task) {
  revng_assert(R.Pipes.count(Task.Name) == 1);
  std::unique_ptr<Pipe> ThePipe = R.Pipes[Task.Name](Task.StaticConfig);
  std::vector<Container *> Containers;
  revng::pypeline::Request Incoming;
  revng::pypeline::Request Outgoing;

  for (const PipeArgs &Arg : Task.Args) {
    std::vector<const ObjectID *> IncomingChunk = pointerize(Arg.Incoming);
    std::vector<const ObjectID *> OutgoingChunk = pointerize(Arg.Outgoing);

    Containers.push_back(&*CMap[Arg.Name]);
    Incoming.push_back(IncomingChunk);
    Outgoing.push_back(OutgoingChunk);
  }

  ThePipe->run(&TheModel, Containers, Incoming, Outgoing, Task.DynamicConfig);
}

static void runTask(RegistryImpl &R,
                    ContainerMap &CMap,
                    Model &TheModel,
                    const AnalysisTask &Task) {
  revng_assert(R.Analyses.count(Task.Name) == 1);
  std::unique_ptr<Analysis> TheAnalysis = R.Analyses[Task.Name]();
  std::vector<Container *> Containers;
  revng::pypeline::Request Incoming;

  for (const AnalysisArgs &Arg : Task.Args) {
    std::vector<const ObjectID *> IncomingChunk = pointerize(Arg.Incoming);

    Containers.push_back(&*CMap[Arg.Name]);
    Incoming.push_back(IncomingChunk);
  }

  llvm::Error Err = TheAnalysis->run(&TheModel,
                                     Containers,
                                     Incoming,
                                     Task.Config);
  revng_assert(not Err);
}

static void
runTask(ContainerMap &CMap, SavePoint &SV, const SavePointTask &Task) {
  for (const SavePointContainer &SPContainer : Task.Containers) {
    Container &Cont = *CMap[SPContainer.Name];
    SV.save(Cont,
            SPContainer.Name,
            Task.ID,
            SPContainer.ConfigurationHash,
            toRef(SPContainer.Incoming));
    SV.load(Cont,
            SPContainer.Name,
            Task.ID,
            SPContainer.ConfigurationHash,
            toRef(SPContainer.Outgoing));
  }
}

void Runner::run(Model &TheModel, const TraceFile &File, SavePoint &SP) {
  ContainerMap Containers;

  for (const ContainerDeclaration &Decl : File.Containers) {
    revng_assert(Registry.Containers.count(Decl.Type) == 1);
    Containers[Decl.Name] = Registry.Containers[Decl.Type]();
  }

  for (const TaskDeclaration &Decl : File.Tasks) {
    if (std::holds_alternative<PipeTask>(Decl)) {
      runTask(Registry, Containers, TheModel, std::get<PipeTask>(Decl));
    } else if (std::holds_alternative<AnalysisTask>(Decl)) {
      runTask(Registry, Containers, TheModel, std::get<AnalysisTask>(Decl));
    } else if (std::holds_alternative<SavePointTask>(Decl)) {
      runTask(Containers, SP, std::get<SavePointTask>(Decl));
    } else {
      revng_abort();
    }
  }
}
