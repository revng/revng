#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/Support/FunctionTags.h"

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
  using LLVMRootContainer = revng::pypeline::LLVMRootContainer;
  static_assert(std::is_same_v<typename Base::OutputContainerType,
                               LLVMRootContainer>);
  using ContainerTypesRef = detail::TupleWithRef<typename Base::ContainerTypes>;

  class Pass : public llvm::ModulePass {
  private:
    ObjectDependenciesHelper &ODH;
    const revng::pypeline::Request &Outgoing;

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
      Containers(Containers) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
      forEach<typename T::Analyses>([&AU]<typename A, size_t I>() {
        AU.addRequired<A>();
      });
    }

    bool runOnModule(llvm::Module &Module) override {
      std::map<MetaAddress, llvm::Function *> AddressToFunction;
      for (llvm::Function &Function : Module.functions()) {
        if (not FunctionTags::Isolated.isTagOf(&Function))
          continue;

        AddressToFunction.emplace(getMetaAddressOfIsolatedFunction(Function),
                                  &Function);
      }

      auto Maker = [&]<size_t... I>(std::index_sequence<I...>) {
        return T{ *this,
                  Model,
                  this->StaticConfiguration,
                  Configuration,
                  std::get<I>(Containers)... };
      };
      T Instance = Maker(std::make_index_sequence<Base::ContainerCount>());

      const model::Binary &Binary = *Model.get().get();
      for (const ObjectID *Object : Outgoing.at(Base::OutputContainerIndex)) {
        auto Committer = ODH.getCommitterFor(*Object,
                                             Base::OutputContainerIndex);

        const MetaAddress &Entry = std::get<MetaAddress>(Object->key());
        const model::Function &Function = Binary.Functions().at(Entry);
        llvm::Function *LLVMFunction = AddressToFunction.at(Function.Entry());
        Instance.runOnFunction(Function, *LLVMFunction);
      }

      return true;
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
  using ArgumentsDocumentation = T::Arguments;

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
    LLVMRootContainer
      &ModuleContainer = std::get<Base::OutputContainerIndex>(ContainersRef);
    Manager.run(ModuleContainer.getModule());

    return ODH.takeDependencies();
  }
};
