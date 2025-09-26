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
#include "revng/Support/TemporaryFile.h"

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
  std::optional<TemporaryFile> File;

public:
  inline static char ID = '0';
  inline static const llvm::StringRef MIMEType = MIME;
  inline static const char *Name = TypeName;

  FileContainer(llvm::StringRef Name) :
    pipeline::Container<FileContainer>(Name), File() {}
  ~FileContainer() override { remove(); }

  FileContainer(const FileContainer &Other) { *this = Other; }
  FileContainer &operator=(const FileContainer &Other) noexcept {
    if (this == &Other)
      return *this;

    if (Other.empty())
      remove();
    else
      cantFail(llvm::sys::fs::copy_file(Other.File->path(), getOrCreatePath()));

    return *this;
  }

  FileContainer(FileContainer &&Other) { *this = std::move(Other); }
  FileContainer &operator=(FileContainer &&Other) noexcept {
    if (this == &Other)
      return *this;

    remove();
    File = std::move(Other.File);
    return *this;
  }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Container) const final {
    auto Result = std::make_unique<FileContainer>(this->name());

    // Return an empty FileContainer if we are empty or our target has not been
    // requested
    if (empty() or not Container.contains(getOnlyPossibleTarget()))
      return Result;

    *Result = *this;
    return Result;
  }

  pipeline::TargetsList enumerate() const final {
    if (empty())
      return {};

    return pipeline::TargetsList({ getOnlyPossibleTarget() });
  }

  bool removeImpl(const pipeline::TargetsList &Target) final {
    auto NotFound = llvm::find(Target, getOnlyPossibleTarget()) == Target.end();
    if (NotFound)
      return false;

    this->clear();

    return true;
  }

  llvm::Error store(const revng::FilePath &Path) const override {
    // We must ensure that if we got invalidated then no file on disk is
    // present, so that the next time we load we don't mistakenly think that we
    // have some content.
    //
    // Other containers do not need this because they determine their content by
    // looking inside the stored file, instead of only checking if the file
    // exists.
    if (empty()) {
      auto MaybeExists = Path.exists();
      if (not MaybeExists)
        return MaybeExists.takeError();

      if (MaybeExists.get())
        return Path.remove();

      return llvm::Error::success();
    }

    return revng::FilePath::fromLocalStorage(File->path()).copyTo(Path);
  }

  llvm::Error loadImpl(const revng::FilePath &Path) override {
    auto MaybeExists = Path.exists();
    if (not MaybeExists)
      return MaybeExists.takeError();

    if (not MaybeExists.get()) {
      this->clear();
      return llvm::Error::success();
    }

    return Path.copyTo(revng::FilePath::fromLocalStorage(getOrCreatePath()));
  }

  void clearImpl() override { *this = FileContainer(this->name()); }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    if (empty())
      return llvm::Error::success();

    if (auto MaybeBuffer = llvm::MemoryBuffer::getFile(File->path()))
      OS << (*MaybeBuffer)->getBuffer();
    else
      return llvm::createStringError(MaybeBuffer.getError(),
                                     "could not read file");

    return llvm::Error::success();
  }

  llvm::Error deserializeImpl(const llvm::MemoryBuffer &Buffer) override {
    std::error_code EC;
    llvm::raw_fd_ostream OS(getOrCreatePath(), EC, llvm::sys::fs::OF_None);
    if (EC)
      return llvm::createStringError(EC,
                                     "could not write file at %s",
                                     File->path().str().c_str());

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
    if (empty())
      return std::nullopt;
    return File->path();
  }

  llvm::StringRef getOrCreatePath() {
    if (empty())
      File.emplace(llvm::Twine("revng-") + this->name(), Suffix);
    return File->path();
  }

  bool empty() const { return not File.has_value(); }
  bool exists() const { return File.has_value(); }

  void dump() const debug_function { dbg << File->path().str() << "\n"; }

private:
  void mergeBackImpl(FileContainer &&Container) override {
    if (Container.exists())
      *this = std::move(Container);
  }

  void remove() { File.reset(); }

  pipeline::Target getOnlyPossibleTarget() const {
    return pipeline::Target({}, *K);
  }
};

inline constexpr char BinaryFileMIMEType[] = "application/"
                                             "x-executable";

inline constexpr char BinaryFileSuffix[] = "";
inline constexpr char BinaryFileName[] = "binary";
using BinaryFileContainer = FileContainer<&kinds::Binary,
                                          BinaryFileName,
                                          BinaryFileMIMEType,
                                          BinaryFileSuffix>;

inline constexpr char ObjectFileMIMEType[] = "application/x-object";
inline constexpr char ObjectFileSuffix[] = ".o";
inline constexpr char ObjectFileName[] = "object";
using ObjectFileContainer = FileContainer<&kinds::Object,
                                          ObjectFileName,
                                          ObjectFileMIMEType,
                                          ObjectFileSuffix>;

inline constexpr char TranslatedFileMIMEType[] = "application/x-executable";
inline constexpr char TranslatedFileSuffix[] = "";
inline constexpr char TranslatedFileName[] = "translated";
using TranslatedFileContainer = FileContainer<&kinds::Translated,
                                              TranslatedFileName,
                                              TranslatedFileMIMEType,
                                              TranslatedFileSuffix>;

inline constexpr char HexDumpFileMIMEType[] = "text/x.hexdump+ptml";
inline constexpr char HexDumpFileSuffix[] = ".hex";
inline constexpr char HexDumpFileName[] = "hex-dump";
using HexDumpFileContainer = FileContainer<&kinds::HexDump,
                                           HexDumpFileName,
                                           HexDumpFileMIMEType,
                                           HexDumpFileSuffix>;

} // namespace revng::pipes
