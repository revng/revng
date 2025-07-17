#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/MemoryBuffer.h"

#include "revng/Pypeline/ObjectID.h"

namespace revng::pypeline::tracerunner {

struct ContainerDeclaration {
  std::string Name;
  std::string Type;
};

struct PipeArgs {
  std::string Name;
  std::vector<ObjectID> Incoming;
  std::vector<ObjectID> Outgoing;
};

struct PipeTask {
  std::string Name;
  std::string StaticConfig;
  std::string DynamicConfig;
  std::vector<PipeArgs> Args;
};

struct AnalysisArgs {
  std::string Name;
  std::vector<ObjectID> Incoming;
};

struct AnalysisTask {
  std::string Name;
  std::string Config;
  std::vector<AnalysisArgs> Args;
};

struct SavePointContainer {
  std::string Name;
  std::string ConfigurationHash;
  std::vector<ObjectID> Incoming;
  std::vector<ObjectID> Outgoing;
};

struct SavePointTask {
  std::string Name;
  uint64_t ID;
  std::vector<SavePointContainer> Containers;
};

using TaskDeclaration = std::variant<PipeTask, AnalysisTask, SavePointTask>;

struct TraceFile {
  std::vector<ContainerDeclaration> Containers;
  std::vector<TaskDeclaration> Tasks;

  static TraceFile deserialize(const llvm::MemoryBuffer &Buffer);
  static llvm::Expected<TraceFile> fromFile(llvm::StringRef Path);
};

} // namespace revng::pypeline::tracerunner
