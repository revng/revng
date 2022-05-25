#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <optional>

#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

namespace revng::pipes {

/// A temporary file is a container that wraps a single temporary file on disk.
/// The enumeration of this container will yield the empty set if there is
/// currently no file associated to a instance of Temporary file, and will
/// return The target ("root", K) otherwise, where K is the kind provided at
/// construction time.
class FileContainer : public pipeline::Container<FileContainer> {
private:
  llvm::SmallString<32> Path;
  pipeline::Kind *K;
  std::string Suffix;

public:
  static char ID;

  FileContainer(pipeline::Kind &K,
                llvm::StringRef Name,
                llvm::StringRef MIMEType,
                llvm::StringRef Suffix);
  FileContainer(FileContainer &&);
  ~FileContainer() override;
  FileContainer(const FileContainer &);
  FileContainer &operator=(const FileContainer &Other) noexcept;
  FileContainer &operator=(FileContainer &&Other) noexcept;

  std::unique_ptr<ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Container) const final;

  pipeline::TargetsList enumerate() const final;

  bool remove(const pipeline::TargetsList &Target) final;

  llvm::Error storeToDisk(llvm::StringRef Path) const override;

  llvm::Error loadFromDisk(llvm::StringRef Path) override;

  void clear() override;
  llvm::Error serialize(llvm::raw_ostream &OS) const override;
  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override;
  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override;

public:
  std::optional<llvm::StringRef> path() const {
    if (Path.empty())
      return std::nullopt;
    return llvm::StringRef(Path);
  }

  llvm::StringRef getOrCreatePath();

  bool exists() const { return not Path.empty(); }

  void dump() const debug_function { dbg << Path.data() << "\n"; }

private:
  void mergeBackImpl(FileContainer &&Container) override;
  void remove();
  pipeline::Target getOnlyPossibleTarget() const {
    return pipeline::Target({}, *K);
  }
};

inline pipeline::ContainerFactory
makeFileContainerFactory(pipeline::Kind &K,
                         llvm::StringRef MIMEType,
                         const llvm::Twine &Suffix = "") {
  std::string SuffixString = Suffix.str();
  return [&K, MIMEType, SuffixString](llvm::StringRef Name) {
    return std::make_unique<FileContainer>(K, Name, MIMEType, SuffixString);
  };
}

} // namespace revng::pipes
