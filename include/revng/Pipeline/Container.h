#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "revng/Pipeline/Target.h"
#include "revng/Support/Concepts.h"

namespace Pipeline {

class ContainerBase {
private:
  const char *ID;
  std::string Name;

public:
  const char *getTypeID() const { return ID; }

  ContainerBase(char const *ID, llvm::StringRef Name) :
    ID(ID), Name(Name.str()) {}

  /// The implementation of cloneFiltered must return a copy of the current
  /// container and a invocation of enumerate on such container must be exactly
  /// equal to Targets
  virtual std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Targets) const = 0;

  /// The implementation of this method must ensure that after the execution
  /// this->enumerate() == before(Other).enumerate().merge(this->enumerate())
  ///
  /// In other words the whole content of Other must be transferred to this.
  virtual void mergeBack(ContainerBase &&Other) = 0;

  /// This method is the method used the pipeline system to understand which
  /// targets are currently available inside the current container.
  virtual TargetsList enumerate() const = 0;

  /// The implementation must ensure that
  /// not after(this)->enumerate().contains(Targets);
  virtual bool remove(const TargetsList &Targets) = 0;

  /// The implementation must ensure that there exists a file at the provided
  /// path that contains the serialized version of this object.
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const = 0;

  /// The  implementation must esure that the content of this file will be
  /// loaded from the provided path.
  virtual llvm::Error loadFromDisk(llvm::StringRef Path) = 0;
  virtual ~ContainerBase() = default;

  static bool classof(const ContainerBase *) { return true; }

  const std::string &name() const { return Name; }
};

/// crtp class to be extended to implement a pipeline container
/// The methods that must be implemented are those shown in containerBase
template<typename Derived>
class Container : public ContainerBase {
public:
  /// This is a template to force the evaluation of the concept from the class
  /// definition to this class objects instantiation, otherwise the derived type
  /// would not be fully defined yet and the constraints would fail.
  template<TypeWithID T = Derived>
  Container(llvm::StringRef Name, const char *ID = &Derived::ID) :
    ContainerBase(ID, Name) {}

  void mergeBack(ContainerBase &&Container) final {
    mergeBackImpl(std::move(llvm::cast<Derived>(Container)));
  }
  ~Container() override = default;

  static bool classof(const ContainerBase *Base) {
    return Base->getTypeID() == &Derived::ID;
  }

  Derived *self() { return static_cast<Derived *>(this); }

  const Derived *self() const { return static_cast<const Derived *>(this); }

protected:
  virtual void mergeBackImpl(Derived &&Container) = 0;
};

} // namespace Pipeline
