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

template<pipeline::SingleElementKind *K,
         const char *TypeName,
         const char *MIME,
         const char *Suffix>
class StringBufferContainer
  : public pipeline::Container<
      StringBufferContainer<K, TypeName, MIME, Suffix>> {
private:
  std::string Content;

public:
  inline static char ID = '0';
  inline static const llvm::StringRef MIMEType = MIME;
  inline static const char *Name = TypeName;

  StringBufferContainer(llvm::StringRef Name,
                        llvm::StringRef StartingContent = "") :
    pipeline::Container<StringBufferContainer>(Name), Content() {}

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Container) const final {
    if (not Container.contains(getOnlyPossibleTarget()))
      return std::make_unique<StringBufferContainer>(this->name());

    return std::make_unique<StringBufferContainer>(*this);
  }

  pipeline::TargetsList enumerate() const final {
    if (Content.empty())
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

  void setContent(std::string NewString) { Content = std::move(NewString); }

  void clearImpl() override { *this = StringBufferContainer(this->name()); }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    OS << Content;
    OS.flush();
    return llvm::Error::success();
  }

  llvm::Error deserializeImpl(const llvm::MemoryBuffer &Buffer) override {
    Content = Buffer.getBuffer().str();
    return llvm::Error::success();
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_check(Target == getOnlyPossibleTarget());
    return serialize(OS);
  }

  llvm::raw_string_ostream asStream() {
    return llvm::raw_string_ostream(Content);
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { K }; }

public:
  void dump() const debug_function { dbg << Content << "\n"; }

private:
  void mergeBackImpl(StringBufferContainer &&Container) override {
    if (Container.Content.empty())
      return;

    Content = std::move(Container.Content);
  }

  pipeline::Target getOnlyPossibleTarget() const {
    return pipeline::Target({}, *K);
  }
};

} // namespace revng::pipes
