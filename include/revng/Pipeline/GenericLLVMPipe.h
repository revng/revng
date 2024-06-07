#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/PassRegistry.h"

#include "revng/ADT/Concepts.h"
#include "revng/Pipeline/ContainerFactorySet.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Step.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ResourceFinder.h"

namespace pipeline {

class LLVMContainer;

class LLVMPassWrapperBase {
public:
  virtual ~LLVMPassWrapperBase() = default;
  virtual void registerPasses(llvm::legacy::PassManager &Manager) = 0;
  virtual const std::vector<ContractGroup> &getContract() const = 0;
  virtual std::unique_ptr<LLVMPassWrapperBase> clone() const = 0;
  virtual llvm::StringRef getName() const = 0;
  virtual void print(llvm::raw_ostream &OS) const = 0;
};

template<typename T>
concept LLVMPass = requires(T P) {
  { T::Name } -> convertible_to<const char *>;
  { P.registerPasses(std::declval<llvm::legacy::PassManager &>()) };
};

template<typename T>
concept LLVMPrintablePass = requires(T P) {
  { P.print(llvm::outs()) };
};

class PureLLVMPassWrapper : public LLVMPassWrapperBase {
private:
  std::string PassName;

public:
  PureLLVMPassWrapper(llvm::StringRef PassName) : PassName(PassName.str()) {}

  static bool passExists(llvm::StringRef PassName) {
    return llvm::PassRegistry::getPassRegistry()->getPassInfo(PassName);
  }

  void print(llvm::raw_ostream &OS) const override { OS << "-" << PassName; }

  static llvm::Expected<std::unique_ptr<PureLLVMPassWrapper>>
  create(llvm::StringRef PassName) {
    if (not passExists(PassName))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not load llvm pass %s ",
                                     PassName.str().c_str());

    return std::make_unique<PureLLVMPassWrapper>(PassName);
  }

  ~PureLLVMPassWrapper() override = default;

  void registerPasses(llvm::legacy::PassManager &Manager) override;

  const std::vector<ContractGroup> &getContract() const override {
    static const std::vector<ContractGroup> Empty{};
    return Empty;
  }
  std::unique_ptr<LLVMPassWrapperBase> clone() const override;

  llvm::StringRef getName() const override { return PassName; }
};

/// LLVM pipes are pipes composed of any number of llvm passes
///
/// The contract is deduced by the used passes, and the passes are assembled
/// into pass managers when needed.
template<LLVMPass T>
class LLVMPassWrapper : public LLVMPassWrapperBase {
public:
  using RegistrationFunctionType = void (*)(llvm::legacy::PassManager &);

private:
  T PipePass;
  std::vector<ContractGroup> Contract;

public:
  LLVMPassWrapper(T Pass) :
    PipePass(std::move(Pass)), Contract(this->PipePass.getContract()) {}

  LLVMPassWrapper(const LLVMPassWrapper &Other) = default;
  LLVMPassWrapper(LLVMPassWrapper &&Other) = default;

  template<typename... ArgsT>
  LLVMPassWrapper(ArgsT &&...Args) :
    PipePass(std::forward<ArgsT>(Args)...),
    Contract(this->PipePass.getContract()) {}

  ~LLVMPassWrapper() override = default;

public:
  llvm::StringRef getName() const override { return T ::Name; }

public:
  void registerPasses(llvm::legacy::PassManager &Manager) override {
    PipePass.registerPasses(Manager);
  }

  const std::vector<ContractGroup> &getContract() const override {
    return Contract;
  }

  std::unique_ptr<LLVMPassWrapperBase> clone() const override {
    return std::make_unique<LLVMPassWrapper>(*this);
  }

  void print(llvm::raw_ostream &OS) const override {
    if constexpr (LLVMPrintablePass<T>)
      PipePass.print(OS);
    else
      OS << "-" << T::Name;
  }
};

/// Implementation of the LLVM pipes to be instantiated for a particular LLVM
/// container
class GenericLLVMPipe {
private:
  llvm::SmallVector<std::unique_ptr<LLVMPassWrapperBase>, 4> Passes;

public:
  static constexpr auto Name = "generic-llvm-pipe";
  template<typename... T>
  explicit GenericLLVMPipe(T... Pass) {
    (addPass(std::move(Pass)), ...);
  }

  GenericLLVMPipe &operator=(const GenericLLVMPipe &Other) {
    if (this == &Other)
      return *this;

    llvm::SmallVector<std::unique_ptr<LLVMPassWrapperBase>, 4> NewPasses;

    for (const auto &P : Other.Passes)
      NewPasses.push_back(P->clone());

    Passes = std::move(NewPasses);
    return *this;
  }

  GenericLLVMPipe(const GenericLLVMPipe &Other) {
    for (const auto &P : Other.Passes)
      Passes.push_back(P->clone());
  }
  GenericLLVMPipe &operator=(GenericLLVMPipe &&Other) = default;
  GenericLLVMPipe(GenericLLVMPipe &&Other) = default;
  ~GenericLLVMPipe() = default;

  std::vector<ContractGroup> getContract() const {
    std::vector<ContractGroup> Contract;
    for (const auto &Element : Passes)
      for (const auto &C : Element->getContract())
        Contract.push_back(C);

    return Contract;
  }

  void run(ExecutionContext &, LLVMContainer &Container);

  void addPass(const PureLLVMPassWrapper &Pass) {
    Passes.emplace_back(Pass.clone());
  }

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  template<LLVMPass T>
  void addPass(T Pass) {
    using Type = LLVMPassWrapper<T>;
    auto Wrapper = std::make_unique<Type>(std::forward<T>(Pass));
    Passes.emplace_back(std::move(Wrapper));
  }

  template<LLVMPass T, typename... ArgsT>
  void emplacePass(ArgsT &&...Args) {
    using Type = LLVMPassWrapper<T>;
    auto Wrapper = std::make_unique<Type>(std::forward<ArgsT>(Args)...);
    Passes.emplace_back(std::move(Wrapper));
  }

  void addPass(std::unique_ptr<LLVMPassWrapperBase> Impl) {
    Passes.emplace_back(std::move(Impl));
  }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    for (const auto &Pass : Passes) {
      indent(OS, Indentation);
      OS << Pass->getName().str() << "\n";
    }
  }

  void dump() const debug_function { dump(dbg); }
};

class O2Pipe {
public:
  static constexpr auto Name = "o2";
  std::vector<ContractGroup> getContract() const { return {}; }

  void registerPasses(llvm::legacy::PassManager &Manager);
};

using LLVMPipe = GenericLLVMPipe;

} // namespace pipeline
