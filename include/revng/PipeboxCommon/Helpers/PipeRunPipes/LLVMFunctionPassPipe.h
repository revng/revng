#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

#include "revng/Model/FunctionTags.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace detail {

template<typename... Args>
inline std::tuple<Args &...> tupleWithRef(TypeList<Args...>);

template<StrictSpecializationOf<TypeList> T>
using TupleWithRef = decltype(tupleWithRef(std::declval<T>()));

inline constexpr llvm::StringRef PassNamePrefix = "LLVMFunctionPassPipe "
                                                  "wrapping PipeRun ";
inline constexpr llvm::StringRef PassArgumentPrefix = "llvm-function-pass-pipe-"
                                                      "for-";

} // namespace detail

template<typename T>
concept IsLLVMFunctionPassPipe = requires(T &PipeRun,
                                          const model::Function &Function,
                                          llvm::Function &LLVMFunction) {
  requires IsMultipleObjectsPipeRun<T>;
  requires hasConstructor<
    T,
    // The Structure of the constructor is:
    // {Pass, Model, StaticConfiguration, Configuration, Containers...}
    concat<
      TypeList<llvm::Pass &, const Model &, llvm::StringRef, llvm::StringRef>,
      ConstructorContainerArguments<T>>>();
  { PipeRun.runOnFunction(Function, LLVMFunction) } -> std::same_as<void>;
};

template<typename T>
class LLVMFunctionPassPipe : public SingleOutputPipeBase<T> {
private:
  using Base = SingleOutputPipeBase<T>;
  static_assert(anyOf<typename Base::OutputContainerType,
                      revng::pypeline::LLVMRootContainer,
                      revng::pypeline::LLVMFunctionContainer>());
  static constexpr bool
    SingleModule = not std::is_same_v<typename Base::OutputContainerType,
                                      revng::pypeline::LLVMFunctionContainer>;
  using ContainerTypesRef = detail::TupleWithRef<typename Base::ContainerTypes>;

  class Pass : public llvm::ModulePass {
  private:
    ObjectDependenciesHelper &ODH;
    const revng::pypeline::Request &Outgoing;
    T PipeRun;

    const Model &Model;
    llvm::StringRef StaticConfiguration;
    llvm::StringRef Configuration;
    ContainerTypesRef &Containers;

  public:
    static inline char ID = 0;

  public:
    Pass(ObjectDependenciesHelper &ODH,
         const revng::pypeline::Request &Outgoing,
         const class Model &Model,
         llvm::StringRef StaticConfiguration,
         llvm::StringRef Configuration,
         ContainerTypesRef &Containers) :
      llvm::ModulePass(ID),

      ODH(ODH),
      Outgoing(Outgoing),

      Model(Model),
      StaticConfiguration(StaticConfiguration),
      Configuration(Configuration),
      Containers(Containers) {

      PipeRun = compile_time::callWithIndexSequence<
        Base::ContainerCount>([&]<size_t... I>() {
        return T{ *this,
                  Model,
                  StaticConfiguration,
                  Configuration,
                  std::get<I>(Containers)... };
      });
    }

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
      forEach<typename T::Analyses>([&AU]<typename A, size_t I>() {
        AU.addRequired<A>();
      });
    }

    bool runOnModule(llvm::Module &Module) override {
      if constexpr (SingleModule)
        return runOnSingleModule(Module);
      else
        return runOnFunctionModule(Module);
    }

    bool runOnSingleModule(llvm::Module &Module) {
      std::map<MetaAddress, llvm::Function *> AddressToFunction;
      for (llvm::Function *Function : getFunctions(Module)) {
        AddressToFunction.emplace(getMetaAddressOfIsolatedFunction(*Function),
                                  Function);
      }

      const model::Binary &Binary = *Model.get().get();
      for (const ObjectID *Object : Outgoing.at(Base::OutputContainerIndex)) {
        auto Committer = ODH.getCommitterFor(*Object,
                                             Base::OutputContainerIndex);

        const MetaAddress &Entry = std::get<MetaAddress>(Object->key());
        const model::Function &Function = Binary.Functions().at(Entry);
        llvm::Function *LLVMFunction = AddressToFunction.at(Function.Entry());
        PipeRun.runOnFunction(Function, *LLVMFunction);
      }

      return true;
    }

    bool runOnFunctionModule(llvm::Module &Module) {
      std::set<llvm::Function *> Functions = getFunctions(Module);
      revng_assert(Functions.size() == 1);
      llvm::Function &LLVMFunction = **Functions.begin();
      MetaAddress Address = getMetaAddressOfIsolatedFunction(LLVMFunction);
      ObjectID Object(Address);
      const model::Binary &Binary = *Model.get().get();

      {
        auto Committer = ODH.getCommitterFor(Object,
                                             Base::OutputContainerIndex);
        const model::Function &Function = Binary.Functions().at(Address);
        PipeRun.runOnFunction(Function, LLVMFunction);
      }
    }

  private:
    static std::set<llvm::Function *> getFunctions(llvm::Module &Module) {
      std::set<llvm::Function *> Result;
      for (llvm::Function &Function : Module.functions()) {
        if (FunctionTags::Isolated.isTagOf(&Function)
            and not Function.isDeclaration())
          Result.insert(&Function);
      }
      return Result;
    }
  };

private:
  static inline std::string
    PassArgument = (detail::PassArgumentPrefix + T::Name).str();
  static inline std::string PassName = (detail::PassNamePrefix + T::Name).str();
  static inline llvm::RegisterPass<Pass> PassRegistrator{ PassArgument,
                                                          PassName,
                                                          true,
                                                          true };

public:
  using Arguments = T::Arguments;

public:
  template<typename... Args>
    requires std::is_same_v<typename Base::ContainerTypes, TypeList<Args...>>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    ObjectDependenciesHelper ODH(Model, Outgoing, this->ContainerCount);

    llvm::legacy::PassManager Manager;
    forEach<typename T::Analyses>([&Manager]<typename A, size_t I>() {
      Manager.add(new A());
    });

    ContainerTypesRef ContainersRef(Containers...);
    Manager.add(new Pass(ODH,
                         Outgoing,
                         Model,
                         this->StaticConfiguration,
                         Configuration,
                         ContainersRef));
    auto &ModuleContainer = std::get<Base::OutputContainerIndex>(ContainersRef);
    if constexpr (SingleModule) {
      Manager.run(ModuleContainer.getModule());
    } else {
      for (const ObjectID *Object : Outgoing[this->OutputContainerIndex]) {
        MetaAddress Address = std::get<MetaAddress>(Object->key());
        Manager.run(ModuleContainer.getModule(Address));
      }
    }

    return ODH.takeDependencies();
  }
};
