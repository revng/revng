#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <system_error>
#include <type_traits>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Target.h"
#include "revng/Storage/Path.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

namespace pipeline {

template<typename T>
concept HasID = requires {
  { T::ID } -> convertible_to<const char &>;
};

class ContainerTypeInfoBase {
public:
  virtual llvm::StringRef getMIMEType() const = 0;
  virtual const char *getID() const = 0;
  virtual ~ContainerTypeInfoBase() = default;
  virtual std::vector<Kind *> getPossibleKinds() const = 0;
};

template<typename T>
class ContainerTypeInfo : public ContainerTypeInfoBase {
public:
  llvm::StringRef getMIMEType() const override { return T::MIMEType; }
  const char *getID() const override { return &T::ID; }
  ~ContainerTypeInfo() override = default;
  std::vector<Kind *> getPossibleKinds() const override {
    return T::possibleKinds();
  }
};

class ContainerBase {
private:
  template<typename Derived>
  friend class Container;

  const char *ID;
  std::string Name;

  using RegistryType = std::vector<std::unique_ptr<ContainerTypeInfoBase>>;
  static RegistryType &getTypeRegistryImpl() {
    static std::vector<std::unique_ptr<ContainerTypeInfoBase>> V;
    return V;
  }

public:
  static const RegistryType &getTypeRegistry() { return getTypeRegistryImpl(); }

  virtual std::vector<Kind *> getPossibleKinds() const = 0;

public:
  ContainerBase(llvm::StringRef Name, char const *ID) :
    ID(ID), Name(Name.str()) {}

public:
  static bool classof(const ContainerBase *) { return true; }

public:
  const char *getTypeID() const { return ID; }
  const std::string &name() const { return Name; }

public:
  virtual ~ContainerBase() = default;

  virtual llvm::StringRef mimeType() const = 0;

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
  virtual llvm::Error store(const revng::FilePath &Path) const;

  /// The implementation must ensure that the content of this file will be
  /// loaded from the provided path.
  virtual llvm::Error load(const revng::FilePath &Path);

  /// Checks that the content of the this container is valid.
  virtual llvm::Error verify() const { return enumerate().verify(*this); }

  /// Return the serialized content of the specified non * target
  virtual llvm::Error extractOne(llvm::raw_ostream &OS,
                                 const Target &Target) const = 0;

public:
  void dump() const debug_function { cantFail(serialize(llvm::dbgs())); }

  void dumpToDisk(const char *Path) const debug_function {
    std::error_code EC;
    llvm::raw_fd_stream Output(Path, EC);
    cantFail(serialize(Output));
  }
};

/// CRTP class to be extended to implement a pipeline container.
///
/// The methods that must be implemented are those shown in ContainerBase.
template<typename Derived>
class Container : public ContainerBase {
private:
  static int registerType() {
    auto &Registry = ContainerBase::getTypeRegistryImpl();
    revng_assert(llvm::find(Registry, &Derived::ID) == Registry.end(),
                 "cannot register same container type twice");
    Registry.push_back(std::make_unique<ContainerTypeInfo<Derived>>());
    return 0;
  }
  inline static const int ForceRegister = registerType();

public:
  /// This is a template to force the evaluation of the concept from the class
  /// definition to this class objects instantiation, otherwise the derived
  /// type would not be fully defined yet and the constraints would fail.
  template<HasID T = Derived>
  Container(llvm::StringRef Name, const char *ID = &Derived::ID) :
    ContainerBase(Name, ID) {}

  ~Container() override = default;

public:
  static bool classof(const ContainerBase *Base) {
    return Base->getTypeID() == &Derived::ID;
  }

  llvm::StringRef mimeType() const override { return Derived::MIMEType; }

  static std::vector<revng::FilePath>
  getWrittenFiles(const revng::FilePath &Path) {
    return { Path };
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
  std::vector<Kind *> getPossibleKinds() const final {
    return Derived::possibleKinds();
  }
};

} // namespace pipeline
