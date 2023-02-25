/// \file ContainerSet.cpp
/// \brief ContainerSets are collections of containers able to spawn them on
/// request

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

llvm::Error ContainerSet::storeToDisk(StringRef Directory) const {
  for (const auto &Pair : Content) {
    llvm::SmallString<128> Filename;
    llvm::sys::path::append(Filename, Directory, Pair.first());
    const auto &Container = Pair.second;
    if (Container == nullptr)
      continue;

    if (auto Error = Container->storeToDisk(Filename); !!Error)
      return Error;
  }
  return Error::success();
}

llvm::Error ContainerSet::loadFromDisk(StringRef Directory) {
  for (auto &Pair : Content) {
    llvm::SmallString<128> Filename;
    llvm::sys::path::append(Filename, Directory, Pair.first());
    if (not llvm::sys::fs::exists(Filename)) {
      Pair.second = nullptr;
      continue;
    }

    if (auto Error = (*this)[Pair.first()].loadFromDisk(Filename); !!Error)
      return Error;
  }
  return Error::success();
}

llvm::Error ContainerSet::verify() const {
  for (const auto &Pair : Content) {
    if (Pair.second == nullptr)
      continue;

    if (auto Error = Pair.second->verify(); Error)
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

llvm::Error ContainerBase::storeToDisk(llvm::StringRef Path) const {
  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::F_None);
  if (EC)
    return llvm::createStringError(EC,
                                   "could not write file at %s",
                                   Path.str().c_str());

  return serialize(OS);
}

llvm::Error ContainerBase::loadFromDisk(llvm::StringRef Path) {
  if (not llvm::sys::fs::exists(Path)) {
    clear();
    return llvm::Error::success();
  }

  if (auto MaybeBuffer = MemoryBuffer::getFile(Path); !MaybeBuffer)
    return llvm::createStringError(MaybeBuffer.getError(),
                                   "could not read file");
  else
    return deserialize(**MaybeBuffer);
}
