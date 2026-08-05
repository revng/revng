#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Native/Pipe.h"
#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/Support/Tar.h"

namespace revng::pypeline::helpers::native::cli {

/// Helper class that allows parsing command-line arguments for containers given
/// a signature object. It also takes care of loading and saving containers from
/// disk; these are assumed to be in the same tar format as the output of
/// `revng project artifact --tar`. This will also take care of storing the
/// containers, these are retrievable with \ref getContainers.
class ContainerHandler {
private:
  using ContainerPath = std::pair<Container *, llvm::StringRef>;

private:
  llvm::ArrayRef<ContainerArgument> Signature;
  std::vector<std::unique_ptr<Container>> Containers;
  std::vector<ContainerPath> ToLoad;
  std::vector<ContainerPath> ToStore;

public:
  ContainerHandler(llvm::ArrayRef<ContainerArgument> Signature) :
    Signature(Signature) {}

  llvm::Error parseCommandline(llvm::ArrayRef<std::string> Arguments) {
    using revng::pypeline::Access;
    using namespace revng::pypeline::helpers::native;

    size_t ExpectedSize = 0;
    for (auto &Argument : Signature)
      ExpectedSize += Argument.Access == Access::ReadWrite ? 2 : 1;

    if (ExpectedSize != Arguments.size()) {
      return revng::createError("Incorrect number of arguments for containers, "
                                "expected %zu got %zu",
                                ExpectedSize,
                                Arguments.size());
    }

    size_t Index = 0;
    for (auto &Argument : Signature) {
      llvm::StringRef Name = Argument.ContainerTypeName;
      revng_assert(Registry.Containers.count(Name) != 0);
      Containers.push_back(Registry.Containers[Name]());
      Container *TheContainer = Containers.back().get();

      if (Argument.Access == Access::Read
          or Argument.Access == Access::ReadWrite) {
        ToLoad.push_back({ TheContainer, Arguments[Index] });
        Index++;
      }

      if (Argument.Access == Access::Write
          or Argument.Access == Access::ReadWrite) {
        ToStore.push_back({ TheContainer, Arguments[Index] });
        Index++;
      }
    }

    return llvm::Error::success();
  }

  llvm::Error loadContainers() {
    for (auto &LoadPair : ToLoad) {
      auto MaybeBuffer = llvm::MemoryBuffer::getFile(LoadPair.second);
      if (not MaybeBuffer)
        return llvm::errorCodeToError(MaybeBuffer.getError());

      TarReader Reader(**MaybeBuffer, TarFormat::Plain);
      for (TarReader::Entry Entry : Reader.entries()) {
        auto MaybeObjectID = ObjectID::deserialize(Entry.Filename);
        if (not MaybeObjectID)
          return MaybeObjectID.takeError();

        std::map<const ObjectID *, llvm::ArrayRef<char>> LoadInput;
        LoadInput[&*MaybeObjectID] = Entry.Data;
        LoadPair.first->deserialize(LoadInput);
      }
    }

    return llvm::Error::success();
  }

  llvm::Error storeContainers() {
    for (auto &StorePair : ToStore) {
      std::error_code EC;
      llvm::raw_fd_ostream OS(StorePair.second, EC);
      if (EC)
        return llvm::errorCodeToError(EC);

      TarWriter Writer(OS, TarFormat::Plain);
      auto Objects = StorePair.first->objects();
      std::vector<ObjectID> ToSave(Objects.begin(), Objects.end());
      auto Serialized = StorePair.first->serialize(ToSave);

      for (auto &[ObjectID, Data] : Serialized) {
        std::string SerializedObjectID = ObjectID.serialize();
        Writer.addMember(SerializedObjectID, Data.data());
      }
    }

    return llvm::Error::success();
  }

  std::vector<Container *> getContainers() {
    std::vector<Container *> Result;
    for (auto &Entry : Containers)
      Result.push_back(Entry.get());
    return Result;
  }
};

} // namespace revng::pypeline::helpers::native::cli
