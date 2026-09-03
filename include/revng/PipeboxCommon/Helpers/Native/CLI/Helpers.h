#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PipeboxCommon/Helpers/Native/Helpers.h"

namespace revng::pypeline::helpers::native::cli {

inline std::string normalizeName(llvm::StringRef Input) {
  Input = Input.trim();
  std::string Result;

  // Strip
  for (const char &C : Input) {
    if (std::isspace(C) or C == '_')
      Result += '-';
    else
      Result += std::tolower(C);
  }

  return Result;
}

inline llvm::Error
parseRequest(llvm::ArrayRef<ContainerArgument> Signature,
             llvm::ArrayRef<std::string> CommandLineObjects,
             std::vector<std::unique_ptr<ObjectID>> &ObjectsPool,
             Request &Outgoing) {
  llvm::StringMap<size_t> ObjectMapping;
  for (auto &&[Index, Argument] : llvm::enumerate(Signature)) {
    std::string Name = normalizeName(Argument.Name);
    ObjectMapping[Name] = Index;
  }

  size_t ObjectsIndex = 0;
  while (ObjectsIndex < CommandLineObjects.size()) {
    llvm::StringRef Name = CommandLineObjects[ObjectsIndex];
    llvm::StringRef Objects = CommandLineObjects[ObjectsIndex + 1];

    if (ObjectMapping.count(Name) == 0)
      return revng::createError("Container %s not found in signature",
                                Name.str().c_str());

    llvm::SmallVector<llvm::StringRef, 0> Parts;
    Objects.split(Parts, ',');
    for (llvm::StringRef Part : Parts) {
      auto MaybeObject = ObjectID::deserialize(Part);
      if (not MaybeObject)
        return MaybeObject.takeError();

      ObjectsPool.push_back(std::make_unique<ObjectID>(*MaybeObject));
      Outgoing[ObjectMapping[Name]].push_back(ObjectsPool.back().get());
    }

    ObjectsIndex += 2;
  }

  return llvm::Error::success();
}

inline llvm::Expected<Model> modelFromPath(llvm::StringRef Path) {
  using llvm::MemoryBuffer;
  auto ModelBuffer = revng::cantFail(MemoryBuffer::getFile(Path));
  auto BufferRef = llvm::arrayRefFromStringRef(ModelBuffer->getBuffer());
  return Model::deserialize(BufferRef, std::nullopt);
}

// The `--configuration` and `--static-configuration` options carry the path to
// a file; an empty path means no configuration.
inline std::string configurationFromPath(llvm::StringRef Path) {
  using llvm::MemoryBuffer;
  if (Path.empty())
    return "";
  auto Buffer = revng::cantFail(MemoryBuffer::getFile(Path));
  return Buffer->getBuffer().str();
}

} // namespace revng::pypeline::helpers::native::cli
