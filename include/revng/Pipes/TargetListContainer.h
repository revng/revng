#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Container.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/TupleTree/TupleTree.h"

namespace detail {

class TargetStorage : public std::vector<std::string> {
public:
  TargetStorage() = default;
  TargetStorage(const std::vector<std::string> Input) { append(Input, *this); }
};

using TargetListStorage = std::vector<TargetStorage>;

} // namespace detail

namespace llvm::yaml {

template<>
struct SequenceTraits<::detail::TargetListStorage> {
  static size_t size(IO &IO, ::detail::TargetListStorage &Seq) {
    return Seq.size();
  }

  static ::detail::TargetStorage &
  element(IO &, ::detail::TargetListStorage &Seq, size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);

    return Seq[Index];
  }
};

template<>
struct SequenceTraits<::detail::TargetStorage> {
  static size_t size(IO &IO, ::detail::TargetStorage &Seq) {
    return Seq.size();
  }

  static std::string &
  element(IO &, ::detail::TargetStorage &Seq, size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);

    return Seq[Index];
  }

  static const bool flow = true;
};

} // namespace llvm::yaml

namespace revng::pipes {

/// This container is a workaround for the current limitation of revng pipeline,
/// which doesn't supply the list of requested targets when running pipes. This
/// container needs to be created dynamically with a name and a Kind, and,
/// together with the PopulateTargetListContainer, will fill the container with
/// a list of targets derived from Kind::appendAllTargets. This container and
/// the related pipe can be dropped once the pipeline provides itself a list of
/// targets that have been requested to be produced.
template<auto *TheKind, const char *TypeName>
  requires std::is_convertible_v<decltype(TheKind), pipeline::Kind *>
class TargetListContainer
  : public pipeline::Container<TargetListContainer<TheKind, TypeName>> {
private:
  std::set<pipeline::Target> Targets;

public:
  inline static char ID = '0';
  static constexpr auto MIMEType = "application/x.target-list";
  inline static const char *Name = TypeName;
  static constexpr pipeline::Kind *Kind = TheKind;

public:
  TargetListContainer(llvm::StringRef Name) :
    pipeline::Container<TargetListContainer>(Name), Targets() {}

  TargetListContainer(const TargetListContainer &) = default;
  TargetListContainer &operator=(const TargetListContainer &) = default;

  TargetListContainer(TargetListContainer &&) = default;
  TargetListContainer &operator=(TargetListContainer &&) = default;

  ~TargetListContainer() override = default;

public:
  void clear() override { Targets.clear(); }

  void fill(const pipeline::Context &Context) {
    clear();

    pipeline::TargetsList List;
    Kind->appendAllTargets(Context, List);

    for (pipeline::Target &Target : List) {
      Targets.insert(Target);
    }
  }

  std::vector<std::reference_wrapper<const pipeline::Target>>
  getTargets() const {
    std::vector<std::reference_wrapper<const pipeline::Target>> Result;
    for (const pipeline::Target &Target : Targets)
      Result.push_back(Target);

    return Result;
  }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override {
    auto Clone = std::make_unique<TargetListContainer>(*this);

    // Drop all the entries that are not in Targets
    std::erase_if(Clone->Targets, [&](const pipeline::Target &Target) {
      return not Targets.contains(Target);
    });

    return Clone;
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_check(&Target.getKind() == Kind);
    if (Targets.contains(Target)) {
      return llvm::Error::success();
    } else {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Target not found");
    }
  }

  pipeline::TargetsList enumerate() const override {
    pipeline::TargetsList::List Result;
    for (const pipeline::Target &Target : Targets)
      Result.push_back(Target);

    return Result;
  }

  bool remove(const pipeline::TargetsList &Targets) override {
    bool Changed = false;

    for (const pipeline::Target &Target : Targets) {
      revng_assert(&Target.getKind() == Kind);

      size_t Removed = this->Targets.erase(Target);
      Changed = Changed or Removed == 1;
    }

    return Changed;
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    ::detail::TargetListStorage Storage;
    for (const pipeline::Target &Target : Targets) {
      Storage.push_back(Target.getPathComponents());
    }

    llvm::yaml::Output YAMLOutput(OS);
    YAMLOutput << Storage;
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override {
    ::detail::TargetListStorage Storage;

    llvm::yaml::Input YAMLInput(Buffer);
    YAMLInput >> Storage;

    std::error_code EC = YAMLInput.error();
    if (EC)
      return llvm::errorCodeToError(EC);

    clear();
    for (const std::vector<std::string> &PathComponents : Storage) {
      Targets.insert({ PathComponents, *Kind });
    }

    return llvm::Error::success();
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { TheKind }; }

protected:
  void mergeBackImpl(TargetListContainer &&Other) override {
    for (const pipeline::Target &Target : Other.Targets) {
      Targets.insert(Target);
    }
  }
};

} // namespace revng::pipes
