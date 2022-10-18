#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <optional>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/Assert.h"

namespace revng::pipes {

/// A temporary file is a container that wraps a single temporary file on disk.
/// The enumeration of this container will yield the empty set if there is
/// currently no file associated to a instance of Temporary file, and will
/// return The target ("root", K) otherwise, where K is the kind provided at
/// construction time.
template<pipeline::SingleElementKind *K,
         const char *TypeName,
         const char *MIME,
         const char *Suffix>
class FileContainer
  : public pipeline::Container<FileContainer<K, TypeName, MIME, Suffix>> {
private:
  llvm::SmallString<32> Path;

  static void cantFail(std::error_code EC) { revng_assert(!EC); }

  static std::unique_ptr<llvm::MemoryBuffer>
  unfailableGetFileAsStream(llvm::StringRef Path) {
    auto Value = errorOrToExpected(llvm::MemoryBuffer::getFileAsStream(Path));
    return llvm::cantFail(std::move(Value));
  }

public:
  inline static char ID = '0';
  inline static const llvm::StringRef MIMEType = MIME;
  inline static const char *Name = TypeName;

  FileContainer(llvm::StringRef Name) :
    pipeline::Container<FileContainer>(Name), Path() {}

  FileContainer(FileContainer &&);
  ~FileContainer() override { remove(); }

  FileContainer(const FileContainer &);
  FileContainer &operator=(const FileContainer &Other) noexcept {
    if (this == &Other)
      return *this;

    if (Path.empty()) {
      using llvm::sys::fs::createTemporaryFile;
      cantFail(createTemporaryFile(llvm::Twine("revng-") + this->name(),
                                   Other.Suffix,
                                   Path));
      llvm::sys::RemoveFileOnSignal(Path);
    }
    cantFail(llvm::sys::fs::copy_file(Other.Path, Path));
    return *this;
  }

  FileContainer &operator=(FileContainer &&Other) noexcept {
    if (this == &Other)
      return *this;

    remove();
    Path = std::move(Other.Path);
    return *this;
  }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Container) const final {
    bool MustCloneFile = Container.contains(getOnlyPossibleTarget());

    if (not MustCloneFile) {
      return std::make_unique<FileContainer>(this->name());
    }

    auto Result = std::make_unique<FileContainer>(this->name());
    Result->getOrCreatePath();
    cantFail(llvm::sys::fs::copy_file(Path, Result->Path));
    return Result;
  }

  pipeline::TargetsList enumerate() const final {
    if (Path.empty())
      return {};

    return pipeline::TargetsList({ getOnlyPossibleTarget() });
  }

  bool remove(const pipeline::TargetsList &Target) final {
    auto NotFound = llvm::find(Target, getOnlyPossibleTarget()) == Target.end();
    if (NotFound)
      return false;

    clear();

    return true;
  }

  llvm::Error storeToDisk(llvm::StringRef Path) const override {
    using namespace llvm;
    // We must ensure that if we got invalidated then no file on disk is
    // present, so that the next time we load we don't mistakenly think that we
    // have some content.
    //
    // Other containers do not need this because they determine their content by
    // looking inside the stored file, instead of only checking if the file
    // exists.
    if (this->Path.empty()) {
      if (llvm::sys::fs::exists(Path))
        llvm::sys::fs::remove(Path);

      return llvm::Error::success();
    }

    if (Path == "-") {
      auto Buffer = unfailableGetFileAsStream(this->Path);

      llvm::outs() << Buffer->getBuffer();
    }

    auto Error = errorCodeToError(llvm::sys::fs::copy_file(this->Path, Path));
    auto MaybeError = llvm::sys::fs::getPermissions(this->Path);
    auto Perm = llvm::cantFail(llvm::errorOrToExpected(std::move(MaybeError)));
    llvm::sys::fs::setPermissions(Path, Perm);
    return Error;
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) override {
    if (not llvm::sys::fs::exists(Path)) {
      *this = FileContainer(this->name());
      return llvm::Error::success();
    }
    getOrCreatePath();
    return llvm::errorCodeToError(llvm::sys::fs::copy_file(Path, this->Path));
  }

  void clear() override { *this = FileContainer(this->name()); }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    if (Path.empty())
      return llvm::Error::success();

    if (auto MaybeBuffer = llvm::MemoryBuffer::getFile(Path); !MaybeBuffer)
      return llvm::createStringError(MaybeBuffer.getError(),
                                     "could not read file");
    else
      OS << (*MaybeBuffer)->getBuffer();

    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override {
    std::error_code EC;
    llvm::raw_fd_ostream OS(getOrCreatePath(), EC, llvm::sys::fs::F_None);
    if (EC)
      return llvm::createStringError(EC,
                                     "could not write file at %s",
                                     Path.str().str().c_str());

    OS << Buffer.getBuffer();
    return llvm::Error::success();
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_check(Target == getOnlyPossibleTarget());
    return serialize(OS);
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { K }; }

public:
  std::optional<llvm::StringRef> path() const {
    if (Path.empty())
      return std::nullopt;
    return llvm::StringRef(Path);
  }

  llvm::StringRef getOrCreatePath() {
    if (Path.empty()) {
      using llvm::sys::fs::createTemporaryFile;
      cantFail(createTemporaryFile(llvm::Twine("revng-") + this->name(),
                                   Suffix,
                                   Path));
      llvm::sys::RemoveFileOnSignal(Path);
    }

    return llvm::StringRef(Path);
  }

  bool exists() const { return not Path.empty(); }

  void dump() const debug_function { dbg << Path.data() << "\n"; }

private:
  void mergeBackImpl(FileContainer &&Container) override {
    if (not Container.exists())
      return;
    cantFail(llvm::sys::fs::rename(*Container.path(), getOrCreatePath()));
    Container.Path = "";
  }

  void remove() {
    if (not Path.empty()) {
      llvm::sys::DontRemoveFileOnSignal(Path);
      cantFail(llvm::sys::fs::remove(Path));
    }
  }

  pipeline::Target getOnlyPossibleTarget() const {
    return pipeline::Target({}, *K);
  }
};

inline constexpr char BinaryFileMIMEType[] = "application/"
                                             "x-executable";

inline constexpr char BinaryFileSuffix[] = "";
inline constexpr char BinaryFileName[] = "Binary";
using BinaryFileContainer = FileContainer<&kinds::Binary,
                                          BinaryFileName,
                                          BinaryFileMIMEType,
                                          BinaryFileSuffix>;

inline constexpr char ObjectFileMIMEType[] = "application/x-object";
inline constexpr char ObjectFileSuffix[] = ".o";
inline constexpr char ObjectFileName[] = "Object";
using ObjectFileContainer = FileContainer<&kinds::Object,
                                          ObjectFileName,
                                          ObjectFileMIMEType,
                                          ObjectFileSuffix>;

inline constexpr char TranslatedFileMIMEType[] = "application/x-executable";
inline constexpr char TranslatedFileSuffix[] = "";
inline constexpr char TranslatedFileName[] = "Translated";
using TranslatedFileContainer = FileContainer<&kinds::Translated,
                                              TranslatedFileName,
                                              TranslatedFileMIMEType,
                                              TranslatedFileSuffix>;

} // namespace revng::pipes
