#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Target.h"

namespace pipeline {

/// We sometime wish to allow a downstream library to extend what Kinds can be
/// contained in a container. An example is a LLVMContainer, it contains a
/// module which is composed of global objects, a downstream library may wish to
/// add a particular meaning of a global object.  If that is the case then
/// ContainerEnumerator and EnumerableContainers are what you need.
///
/// If a downstream user of revng wishes to extend the enumeration and removal
/// rule from a container it can do so by extending this class, specialized with
/// the type of the container in question.
///
/// The enumerate and removeTarget methods must be implemented with the
/// appropriate rules, that is, enumerate must return Targets with the same kind
/// as the one pointed by K.
template<typename Container>
class ContainerEnumerator {
private:
  Kind *K;

public:
  ContainerEnumerator(Kind &K) : K(&K) {
    Container::getRegisteredInspectors().push_back(this);
  }

public:
  Kind &getKind() const { return *K; }

public:
  virtual ~ContainerEnumerator() = default;

  virtual TargetsList enumerate(const Context &Context,
                                const Container &ToInspect) const = 0;

  /// \return must return true if it was possible to remove the provided target
  virtual bool remove(const Context &Context,
                      const TargetsList &Targets,
                      Container &ToInspect) const = 0;
};

/// An inspectable container is a container that delegates the enumeration and
/// removal of its contents to inspectors, which can be dynamically registered
/// at runtime.
///
/// This facility is needed to be able to decouple the Kind declaration from the
/// containers declaration, since if it was always containers to name which
/// kinds can be contained in them, then they would not be able to name kinds
/// declared downstream libraries. Similarly, should be always kind to be able
/// to enumerate the context of a container, then any containers declared in
/// downstream libraries would not be able to use kinds declared upstream.
///
/// You should declare a new inspectable container if and only if you think that
/// downstream user may wish to extend how the enumeration of this container
/// happens.
/// Otherwise simply declare a new Container Type, the difference is that a
/// normal container must provide a enumeration method of its own rather than
/// leaving it up to inspectors.
template<typename Derived>
class EnumerableContainer : public Container<Derived> {
  friend class ContainerEnumerator<Derived>;

private:
  using StaticContainer = llvm::SmallVector<ContainerEnumerator<Derived> *, 4>;
  static StaticContainer &getRegisteredInspectors() {
    static StaticContainer List;
    return List;
  }

protected:
  Context *TheContext;

public:
  EnumerableContainer(Context &Context, llvm::StringRef Name) :
    Container<Derived>(Name), TheContext(&Context) {}

  EnumerableContainer(Context &Context, llvm::StringRef Name, const char *ID) :
    Container<Derived>(Name, ID), TheContext(&Context) {}

public:
  const Context &getContext() const { return *TheContext; }
  Context &getContext() { return *TheContext; }

  bool contains(const Target &Target) const {
    return enumerate().contains(Target);
  }

public:
  ~EnumerableContainer() override = default;

  TargetsList enumerate() const override {
    TargetsList ToReturn;
    for (const auto *Inspector : getRegisteredInspectors())
      ToReturn.merge(Inspector->enumerate(*TheContext, *this->self()));

    return ToReturn;
  }

  /// \return true if all targets to remove have been removed
  bool remove(const TargetsList &Targets) override {
    bool RemovedAll = true;
    for (const auto *Inspector : getRegisteredInspectors()) {
      RemovedAll = Inspector->remove(*TheContext,
                                     Targets.filter(Inspector->getKind()),
                                     *this->self())
                   and RemovedAll;
    }

    return RemovedAll;
  }

  static std::vector<Kind *> possibleKinds() {
    std::vector<Kind *> ToReturn;
    for (auto *Inspector : getRegisteredInspectors()) {
      ToReturn.push_back(&Inspector->getKind());
    }

    return ToReturn;
  }
};

template<typename Container>
class KindForContainer : public Kind, public ContainerEnumerator<Container> {
public:
  template<RankSpecialization BaseRank>
  KindForContainer(llvm::StringRef Name, const BaseRank &Rank) :
    Kind(Name, Rank, {}, {}),
    ContainerEnumerator<Container>(*static_cast<Kind *>(this)) {}

  template<RankSpecialization BaseRank>
  KindForContainer(llvm::StringRef Name, Kind &Parent, const BaseRank &Rank) :
    Kind(Name, Parent, Rank, {}, {}),
    ContainerEnumerator<Container>(*static_cast<Kind *>(this)) {}

public:
  ~KindForContainer() override = default;
};

} // namespace pipeline
