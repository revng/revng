//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/TraceRunner/TraceFile.h"

using namespace revng::pypeline::tracerunner;

LLVM_YAML_IS_SEQUENCE_VECTOR(TaskDeclaration);
LLVM_YAML_IS_SEQUENCE_VECTOR(PipeArgs);
LLVM_YAML_IS_SEQUENCE_VECTOR(AnalysisArgs);
LLVM_YAML_IS_SEQUENCE_VECTOR(SavePointContainer);
LLVM_YAML_IS_SEQUENCE_VECTOR(ContainerDeclaration);
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(ObjectID);

namespace llvm::yaml {

template<>
struct ScalarTraits<ObjectID> {
  static void output(const ObjectID &, void *, llvm::raw_ostream &) {
    revng_abort();
  }

  static StringRef input(StringRef Scalar, void *, ObjectID &Value) {
    Value = llvm::cantFail(ObjectID::deserialize(Scalar));
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};

template<>
struct MappingTraits<ContainerDeclaration> {
  static void mapping(IO &TheIO, ContainerDeclaration &Decl) {
    TheIO.mapRequired("name", Decl.Name);
    TheIO.mapRequired("type", Decl.Type);
  };
};

template<>
struct MappingTraits<PipeArgs> {
  static void mapping(IO &TheIO, PipeArgs &Args) {
    TheIO.mapRequired("name", Args.Name);
    TheIO.mapRequired("incoming", Args.Incoming);
    TheIO.mapRequired("outgoing", Args.Outgoing);
  };
};

template<>
struct MappingTraits<AnalysisArgs> {
  static void mapping(IO &TheIO, AnalysisArgs &Args) {
    TheIO.mapRequired("name", Args.Name);
    TheIO.mapRequired("incoming", Args.Incoming);
  };
};

template<>
struct MappingTraits<SavePointContainer> {
  static void mapping(IO &TheIO, SavePointContainer &Container) {
    TheIO.mapRequired("name", Container.Name);
    TheIO.mapRequired("configuration_hash", Container.ConfigurationHash);
    TheIO.mapRequired("incoming", Container.Incoming);
    TheIO.mapRequired("outgoing", Container.Outgoing);
  };
};

template<>
struct MappingTraits<TaskDeclaration> {
  static void mapping(IO &TheIO, TaskDeclaration &Task) {
    revng_assert(not TheIO.outputting());
    std::string Type;
    TheIO.mapRequired("type", Type);

    if (Type == "Pipe") {
      PipeTask TheTask;
      TheIO.mapRequired("name", TheTask.Name);
      TheIO.mapRequired("static_config", TheTask.StaticConfig);
      TheIO.mapRequired("dynamic_config", TheTask.DynamicConfig);
      TheIO.mapRequired("args", TheTask.Args);
      Task = TheTask;
    } else if (Type == "Analysis") {
      AnalysisTask TheTask;
      TheIO.mapRequired("name", TheTask.Name);
      TheIO.mapRequired("config", TheTask.Config);
      TheIO.mapRequired("args", TheTask.Args);
      Task = TheTask;
    } else if (Type == "Savepoint") {
      SavePointTask TheTask;
      TheIO.mapRequired("name", TheTask.Name);
      TheIO.mapRequired("id", TheTask.ID);
      TheIO.mapRequired("containers", TheTask.Containers);
      Task = TheTask;
    } else {
      revng_abort();
    }
  }
};

template<>
struct llvm::yaml::MappingTraits<TraceFile> {
  static void mapping(IO &TheIO, TraceFile &Trace) {
    TheIO.mapRequired("containers", Trace.Containers);
    TheIO.mapRequired("tasks", Trace.Tasks);
  }
};

} // namespace llvm::yaml

namespace revng::pypeline::tracerunner {

TraceFile TraceFile::deserialize(const llvm::MemoryBuffer &Buffer) {
  llvm::yaml::Input YAMLReader(Buffer);
  TraceFile Result;
  YAMLReader >> Result;
  return Result;
}

llvm::Expected<TraceFile> TraceFile::fromFile(llvm::StringRef Path) {
  auto MaybeInputBuffer = llvm::MemoryBuffer::getFileAsStream(Path);
  if (std::error_code EC = MaybeInputBuffer.getError()) {
    return llvm::createStringError(EC,
                                   "Unable to read input trace: "
                                     + EC.message());
  }
  return TraceFile::deserialize(**MaybeInputBuffer);
}

} // namespace revng::pypeline::tracerunner
