/// \file TemporariFile.cpp
/// \brief used to operate on file on disk.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "revng/Pipes/FileContainer.h"

using namespace llvm;
using namespace llvm::sys;
using namespace pipeline;

static void cantFail(std::error_code EC) {
  revng_assert(!EC);
}

namespace revng::pipes {

char FileContainer::ID;

FileContainer::FileContainer(Kind &K, llvm::StringRef Name) :
  Container<FileContainer>(Name), Path(), K(&K) {
}

FileContainer::~FileContainer() {
  remove();
}

llvm::StringRef FileContainer::getOrCreatePath() {
  if (Path.empty())
    cantFail(llvm::sys::fs::createTemporaryFile("", "", Path));

  return llvm::StringRef(Path);
}

void FileContainer::remove() {
  if (not Path.empty())
    cantFail(llvm::sys::fs::remove(Path));
}

FileContainer &FileContainer::operator=(const FileContainer &Other) noexcept {
  if (this == &Other)
    return *this;

  if (Path.empty())
    cantFail(llvm::sys::fs::createTemporaryFile("", "", Path));
  cantFail(llvm::sys::fs::copy_file(Other.Path, Path));
  return *this;
}

FileContainer &FileContainer::operator=(FileContainer &&Other) noexcept {
  if (this == &Other)
    return *this;

  remove();
  Path = std::move(Other.Path);
  return *this;
}

std::unique_ptr<ContainerBase>
FileContainer::cloneFiltered(const TargetsList &Container) const {
  bool MustCloneFile = Container.contains(getOnlyPossibleTarget());

  if (not MustCloneFile) {
    return std::make_unique<FileContainer>(*K, this->name());
  }

  auto Result = std::make_unique<FileContainer>(*K, this->name());
  Result->getOrCreatePath();
  cantFail(llvm::sys::fs::copy_file(Path, Result->Path));
  return Result;
}

void FileContainer::mergeBackImpl(FileContainer &&Container) {
  if (not Container.exists())
    return;
  cantFail(llvm::sys::fs::rename(*Container.path(), getOrCreatePath()));
  Container.Path = "";
}

llvm::Error FileContainer::storeToDisk(llvm::StringRef Path) const {
  if (this->Path.empty())
    return llvm::Error::success();

  auto Error = errorCodeToError(llvm::sys::fs::copy_file(this->Path, Path));
  auto Perm = cantFail(errorOrToExpected(fs::getPermissions(this->Path)));
  fs::setPermissions(Path, Perm);
  return Error;
}

llvm::Error FileContainer::loadFromDisk(llvm::StringRef Path) {
  if (not llvm::sys::fs::exists(Path)) {
    *this = FileContainer(*K, this->name());
    return llvm::Error::success();
  }
  getOrCreatePath();
  return errorCodeToError(llvm::sys::fs::copy_file(Path, this->Path));
}

TargetsList FileContainer::enumerate() const {
  if (Path.empty())
    return {};

  return TargetsList({ getOnlyPossibleTarget() });
}
} // namespace revng::pipes
