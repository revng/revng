/// \file ContainerSet.cpp
/// ContainerSets are collections of containers able to spawn them on request.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Errors.h"

using namespace pipeline;
using namespace llvm;
using namespace std;

ContainerSet ContainerSet::cloneFiltered(const ContainerToTargetsMap &Targets) {
  ContainerSet ToReturn;
  for (const auto &Pair : Content) {
    const auto &ContainerName = Pair.first();
    const auto &Container = Pair.second;

    auto ExtractedNames = Targets.contains(ContainerName) ?
                            Targets.at(ContainerName) :
                            TargetsList();

    auto Cloned = Container != nullptr ?
                    Container->cloneFiltered(ExtractedNames) :
                    nullptr;

    ToReturn.add(ContainerName, *Factories[Pair.first()], std::move(Cloned));
  }
  revng_assert(ToReturn.Content.size() == Content.size());
  return ToReturn;
}

bool ContainerSet::contains(const Target &Target) const {
  return llvm::any_of(Content, [&Target](const auto &Container) {
    return Container.second->enumerate().contains(Target);
  });
}

Error ContainerSet::remove(const ContainerToTargetsMap &ToRemove) {
  for (const auto &Target : ToRemove) {
    const auto &ContainerName = Target.first();
    const auto &NamesToRemove = Target.second;

    if (not contains(ContainerName))
      continue;

    if (NamesToRemove.size() == 0)
      continue;
    if (auto Ok = at(ContainerName).remove(NamesToRemove); not Ok)
      return make_error<UnknownTargetError>(NamesToRemove, ContainerName);
  }
  return Error::success();
}

void ContainerSet::intersect(ContainerToTargetsMap &ToIntersect) const {
  for (auto &ContainerStatus : ToIntersect) {
    const auto &ContainerName = ContainerStatus.first();
    auto &Names = ContainerStatus.second;

    if (not contains(ContainerName)) {
      Names = {};
      continue;
    }

    const auto &Container = at(ContainerName);
    auto Enumerated = Container.enumerate();
    erase_if(Names, [&Enumerated](const Target &Target) {
      return not Enumerated.contains(Target);
    });
  }
}

llvm::Error ContainerSet::store(const revng::DirectoryPath &Directory) const {
  for (const auto &Pair : Content) {
    revng::FilePath Filename = Directory.getFile(Pair.first());
    const auto &Container = Pair.second;
    if (Container == nullptr)
      continue;

    if (not Container->isDirty())
      continue;

    if (auto Error = Container->store(Filename); !!Error)
      return Error;

    Container->resetDirtiness();
  }
  return Error::success();
}

llvm::Error ContainerSet::load(const revng::DirectoryPath &Directory) {
  for (auto &Pair : Content) {
    revng::FilePath Filename = Directory.getFile(Pair.first());
    auto MaybeExists = Filename.exists();
    if (!MaybeExists)
      return MaybeExists.takeError();

    if (not MaybeExists.get()) {
      Pair.second = nullptr;
      continue;
    }

    if (auto Error = (*this)[Pair.first()].load(Filename); !!Error)
      return Error;
  }
  return Error::success();
}

std::vector<revng::FilePath>
ContainerSet::getWrittenFiles(const revng::DirectoryPath &Directory) const {
  std::vector<revng::FilePath> Result;

  for (const auto &Pair : Factories) {
    revng::FilePath Filename = Directory.getFile(Pair.first());
    append(Pair.second->getWrittenFiles(Filename), Result);
  }

  return Result;
}

llvm::Error ContainerSet::verify() const {
  for (const auto &Pair : Content) {
    if (Pair.second == nullptr)
      continue;

    if (auto Error = Pair.second->verify())
      return Error;
  }
  return llvm::Error::success();
}

ContainerToTargetsMap ContainerSet::enumerate() const {
  ContainerToTargetsMap Status;

  for (const auto &Pair : *this) {
    const auto &Name = Pair.first();
    const auto &MaybeCont = Pair.second;

    if (MaybeCont != nullptr)
      Status[Name] = MaybeCont->enumerate();
  }
  return Status;
}

llvm::Error ContainerBase::store(const revng::FilePath &Path) const {
  auto MaybeWritableFile = Path.getWritableFile();
  if (not MaybeWritableFile) {
    return MaybeWritableFile.takeError();
  }

  if (auto Error = serialize(MaybeWritableFile.get()->os()))
    return Error;

  return MaybeWritableFile.get()->commit();
}

llvm::Error ContainerBase::loadImpl(const revng::FilePath &Path) {
  auto MaybeExists = Path.exists();
  if (not MaybeExists)
    return MaybeExists.takeError();

  if (not MaybeExists.get()) {
    clear();
    return llvm::Error::success();
  }

  auto MaybeBuffer = Path.getReadableFile();
  if (not MaybeBuffer)
    return MaybeBuffer.takeError();

  auto &Buffer = MaybeBuffer.get();
  auto Error = deserialize(Buffer->buffer());
  return Error;
}
