#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pipes {

template<TupleTreeCompatible T,
         pipeline::SingleElementKind *K,
         const char *TypeName,
         const char *MIME>
class TupleTreeContainer
  : public pipeline::Container<TupleTreeContainer<T, K, TypeName, MIME>> {
private:
  std::optional<TupleTree<T>> Content;

public:
  inline static const char ID = 0;
  inline static const llvm::StringRef MIMEType = MIME;
  inline static const char *Name = TypeName;

  using Base = pipeline::Container<TupleTreeContainer<T, K, TypeName, MIME>>;

  TupleTreeContainer(llvm::StringRef Name) :
    pipeline::Container<TupleTreeContainer>(Name), Content(std::nullopt) {}

  TupleTreeContainer(const TupleTreeContainer &Other) :
    TupleTreeContainer(Other.name()) {
    Content = Other.Content;
  }

  TupleTreeContainer(TupleTreeContainer &&Other) :
    TupleTreeContainer(Other.name()) {
    Content = std::move(Other.Content);
  }

  bool empty() const { return not Content.has_value(); }

  const TupleTree<T> &get() const {
    revng_assert(not empty());
    return *Content;
  }

  void insert(const TupleTree<T> &NewTree) { Content = NewTree; }
  void insert(TupleTree<T> &&NewTree) { Content = std::move(NewTree); }

  template<typename... ArgT>
  void emplace(ArgT &&...Args) {
    Content = TupleTree<T>();
    *Content.value() = T(std::forward<ArgT>(Args)...);
  }

  TupleTree<T> &get() {
    revng_assert(not empty());
    return *Content;
  }

  TupleTreeContainer &operator=(const TupleTreeContainer &Other) {
    if (this == &Other)
      return *this;

    Content = Other.Content;
    return *this;
  }

  TupleTreeContainer &operator=(TupleTreeContainer &&Other) {
    if (this == &Other)
      return *this;

    Content = std::move(Other.Content);
    return *this;
  }

  ~TupleTreeContainer() override = default;

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Container) const final {
    if (not Container.contains(pipeline::Target(*K)))
      return std::make_unique<TupleTreeContainer>(this->name());

    return std::make_unique<TupleTreeContainer>(*this);
  }

  pipeline::TargetsList enumerate() const final {
    if (Content)
      return pipeline::TargetsList({ pipeline::Target(*K) });
    return {};
  }

  bool remove(const pipeline::TargetsList &Target) final {
    if (enumerate().contains(Target)) {
      clear();
      return true;
    }
    return false;
  }

  void mergeBackImpl(TupleTreeContainer &&Container) override {
    *this = std::move(Container);
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    if (not Content)
      return llvm::Error::success();

    Content.value().serialize(OS);
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override {
    if (Buffer.getBufferSize() == 0) {
      *this = TupleTreeContainer(this->name());
      return llvm::Error::success();
    }

    const auto &Serialized = Buffer.getBuffer();
    auto Result(llvm::errorOrToExpected(TupleTree<T>::deserialize(Serialized)));
    if (not Result)
      return Result.takeError();

    Content = std::move(*Result);
    return llvm::Error::success();
  }

  llvm::Error storeToDisk(const revng::FilePath &Path) const override {
    using namespace llvm;
    // Tuple tree deserializer does not work on empty files, so we must ensure
    // to not write them
    if (empty()) {
      auto MaybeExists = Path.exists();
      if (not MaybeExists)
        return MaybeExists.takeError();

      if (MaybeExists.get())
        if (auto Error = Path.remove(); Error)
          return Error;

      return llvm::Error::success();
    }

    return Base::storeToDisk(Path);
  }

  llvm::Error loadFromDisk(const revng::FilePath &Path) override {
    auto MaybeExists = Path.exists();
    if (not MaybeExists)
      return MaybeExists.takeError();

    if (not MaybeExists.get()) {
      *this = TupleTreeContainer(this->name());
      return llvm::Error::success();
    }
    return Base::loadFromDisk(Path);
  }

  void clear() override { Content = TupleTree<T>(); }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    if (enumerate().contains(Target))
      return serialize(OS);
    else
      TupleTree<T>().serialize(OS);
    return llvm::Error::success();
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { K }; }
};

} // namespace revng::pipes
