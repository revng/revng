/// \file TemporariFile.cpp
/// \brief used to operate on file on disk.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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

FileContainer::FileContainer(Kind &K,
                             llvm::StringRef Name,
                             llvm::StringRef Mime,
                             llvm::StringRef Suffix) :
  Container<FileContainer>(Name, Mime.data()),
  Path(),
  K(&K),
  Suffix(Suffix.str()) {
}

FileContainer::~FileContainer() {
  remove();
}

llvm::StringRef FileContainer::getOrCreatePath() {
  if (Path.empty())
    cantFail(llvm::sys::fs::createTemporaryFile("", Suffix, Path));

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
    cantFail(llvm::sys::fs::createTemporaryFile("", Other.Suffix, Path));
  cantFail(llvm::sys::fs::copy_file(Other.Path, Path));
  return *this;
}

bool FileContainer::remove(const pipeline::TargetsList &Target) {
  auto NotFound = llvm::find(Target, getOnlyPossibleTarget()) == Target.end();
  if (NotFound)
    return false;

  clear();

  return true;
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
    return std::make_unique<FileContainer>(*K,
                                           this->name(),
                                           this->mime(),
                                           Suffix);
  }

  auto Result = std::make_unique<FileContainer>(*K,
                                                this->name(),
                                                this->mime(),
                                                Suffix);
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
    *this = FileContainer(*K, this->name(), this->mime(), Suffix);
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

void FileContainer::clear() {
  *this = FileContainer(*K, name(), mime(), Suffix);
}

llvm::Error FileContainer::serialize(llvm::raw_ostream &OS) const {
  if (Path.empty())
    return llvm::Error::success();

  if (auto MaybeBuffer = MemoryBuffer::getFile(Path); !MaybeBuffer)
    return llvm::createStringError(MaybeBuffer.getError(),
                                   "could not read file");
  else
    OS << (*MaybeBuffer)->getBuffer();

  return llvm::Error::success();
}

llvm::Error FileContainer::deserialize(const llvm::MemoryBuffer &Buffer) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(getOrCreatePath(), EC, llvm::sys::fs::F_None);
  if (EC)
    return llvm::createStringError(EC,
                                   "could not write file at %s",
                                   Path.str().str().c_str());

  OS << Buffer.getBuffer();
  return llvm::Error::success();
}

} // namespace revng::pipes
