//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/TraceRunner/Registry.h"
#include "revng/Pypeline/TraceRunner/Runner.h"
#include "revng/Pypeline/TraceRunner/Savepoint.h"

using namespace pypeline::tracerunner;
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

static void runTask(Registry &R,
                    ContainerMap &CMap,
                    const Model &TheModel,
                    const PipeTask &Task) {
  revng_assert(R.Pipes.count(Task.Name) == 1);
  std::unique_ptr<Pipe> ThePipe = R.Pipes[Task.Name](Task.StaticConfig);
  std::vector<Container *> Containers;
  pypeline::RequestT Incoming;
  pypeline::RequestT Outgoing;

  for (const PipeArgs &Arg : Task.Args) {
    std::vector<const ObjectID *> IncomingChunk = pointerize(Arg.Incoming);
    std::vector<const ObjectID *> OutgoingChunk = pointerize(Arg.Outgoing);

    Containers.push_back(&*CMap[Arg.Name]);
    Incoming.push_back(IncomingChunk);
    Outgoing.push_back(OutgoingChunk);
  }

  ThePipe->run(&TheModel, Containers, Incoming, Outgoing, Task.DynamicConfig);
}

static void runTask(Registry &R,
                    ContainerMap &CMap,
                    Model &TheModel,
                    const AnalysisTask &Task) {
  revng_assert(R.Analyses.count(Task.Name) == 1);
  std::unique_ptr<Analysis> TheAnalysis = R.Analyses[Task.Name]();
  std::vector<Container *> Containers;
  pypeline::RequestT Incoming;

  for (const AnalysisArgs &Arg : Task.Args) {
    std::vector<const ObjectID *> IncomingChunk = pointerize(Arg.Incoming);

    Containers.push_back(&*CMap[Arg.Name]);
    Incoming.push_back(IncomingChunk);
  }

  TheAnalysis->run(&TheModel, Containers, Incoming, Task.Config);
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

void pypeline::tracerunner::Runner::run(Model &TheModel,
                                        const TraceFile &File,
                                        SavePoint &SP) {
  pypeline::tracerunner::Registry Registry;
  TheRegistry.callAll(Registry);

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
