#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/Pipeline/Target.h"

namespace pipeline {

template<typename T>
concept HasID = requires {
  { T::ID } -> convertible_to<const char &>;
};

class ContainerBase {
private:
  const char *ID;
  std::string Name;
  std::string MIMEType;

public:
  ContainerBase(char const *ID,
                llvm::StringRef Name,
                llvm::StringRef MIMEType) :
    ID(ID), Name(Name.str()), MIMEType(MIMEType.str()) {}

public:
  static bool classof(const ContainerBase *) { return true; }

public:
  const char *getTypeID() const { return ID; }
  const std::string &name() const { return Name; }
  const std::string &mimeType() const { return MIMEType; }

public:
  virtual ~ContainerBase() = default;

  /// The implementation of cloneFiltered must return a copy of the current
  /// container and a invocation of enumerate on such container must be
  /// exactly equal to Targets
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
  ///
  /// returns false if nothing was removed
  virtual bool remove(const TargetsList &Targets) = 0;

  /// The implementation for a Type T that extends ContainerBase must ensure
  /// that a new instance of T, on which deserialize has been invoked with the
  /// serialized content of a old instance, must be equal to the old instance.
  virtual llvm::Error serialize(llvm::raw_ostream &OS) const = 0;

  /// same as serialize
  virtual llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) = 0;

  /// Must reset the state of the container to the just built state
  virtual void clear() = 0;

  /// The implementation must ensure that there exists a file at the provided
  /// path that contains the serialized version of this object.
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const;

  /// The implementation must ensure that the content of this file will be
  /// loaded from the provided path.
  virtual llvm::Error loadFromDisk(llvm::StringRef Path);

  /// Checks that the content of the this container is valid.
  virtual llvm::Error verify() const { return enumerate().verify(*this); }

  /// Return the serialized content of the specified non * target
  virtual llvm::Error
  extractOne(llvm::raw_ostream &OS, const Target &Target) const = 0;
};

/// CRTP class to be extended to implement a pipeline container.
///
/// The methods that must be implemented are those shown in ContainerBase.
template<typename Derived>
class Container : public ContainerBase {
public:
  /// This is a template to force the evaluation of the concept from the class
  /// definition to this class objects instantiation, otherwise the derived
  /// type would not be fully defined yet and the constraints would fail.
  template<HasID T = Derived>
  Container(llvm::StringRef Name,
            const llvm::StringRef MIMEType,
            const char *ID = &Derived::ID) :
    ContainerBase(ID, Name, MIMEType) {}

  ~Container() override = default;

public:
  static bool classof(const ContainerBase *Base) {
    return Base->getTypeID() == &Derived::ID;
  }

public:
  void mergeBack(ContainerBase &&Container) final {
    mergeBackImpl(std::move(llvm::cast<Derived>(Container)));
  }

public:
  Derived *self() { return static_cast<Derived *>(this); }
  const Derived *self() const { return static_cast<const Derived *>(this); }

protected:
  virtual void mergeBackImpl(Derived &&Container) = 0;
};

} // namespace pipeline
