#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/Support/FunctionTags.h"

namespace detail {

class ObjectDependeciesHelper {
private:
  const model::Binary &Binary;
  const revng::pypeline::Request &Outgoing;
  revng::pypeline::ObjectDependencies Dependencies;
  bool DependenciesTaken = false;

public:
  ObjectDependeciesHelper(const Model &Model,
                          const revng::pypeline::Request &Outgoing,
                          size_t DependenciesSize) :
    Binary(*Model.get().get()),
    Outgoing(Outgoing),
    Dependencies(DependenciesSize) {
    revng::Tracking::clearAndResume(Binary);
  }
  ~ObjectDependeciesHelper() { revng_assert(DependenciesTaken); }

  ObjectDependeciesHelper(const ObjectDependeciesHelper &) = delete;
  ObjectDependeciesHelper &operator=(const ObjectDependeciesHelper &) = delete;
  ObjectDependeciesHelper(ObjectDependeciesHelper &&) = delete;
  ObjectDependeciesHelper &operator=(ObjectDependeciesHelper &&) = delete;

  void commit(const ObjectID &Object, size_t Index) {
    revng_assert(Index < Dependencies.size());
    std::vector<std::string> ReadPaths = getReadPaths();
    for (std::string &Path : ReadPaths)
      Dependencies[Index].push_back({ Object, Path });
  }

  void commitAllFor(size_t Index) {
    revng_assert(Index < Dependencies.size());
    std::vector<std::string> ReadPaths = getReadPaths();
    for (const ObjectID *Object : Outgoing[Index])
      for (std::string &Path : ReadPaths)
        Dependencies[Index].push_back({ *Object, Path });
  }

  void commitUniqueTarget(size_t Index) {
    revng_assert(Index < Dependencies.size());
    revng_assert(Outgoing[Index].size() == 1);
    commitAllFor(Index);
  }

  void pushReadFields() { revng::Tracking::push(Binary); }

  void popReadFields() { revng::Tracking::pop(Binary); }

  revng::pypeline::ObjectDependencies &&takeDependencies() {
    DependenciesTaken = true;
    return std::move(Dependencies);
  }

private:
  std::vector<std::string> getReadPaths() {
    std::vector<std::string> Result;
    ReadFields ReadFieldsStruct = revng::Tracking::collect(Binary);
    for (const TupleTreePath &Path : ReadFieldsStruct.Read)
      Result.push_back(*pathAsString<model::Binary>(Path));
    for (const TupleTreePath &Path : ReadFieldsStruct.ExactVectors)
      Result.push_back(*pathAsString<model::Binary>(Path));
    return Result;
  }
};

template<typename T, typename T2>
struct SingleOutputPipeTraits {};

template<typename T, typename... Args>
struct SingleOutputPipeTraits<
  T,
  void (*)(const Model &, llvm::StringRef, llvm::StringRef, Args...)> {
  using ContainerTypes = TypeList<Args...>;
  static constexpr std::array
    ArgumentsDocumentation = std::to_array(T::ArgumentsDocumentation);
};

template<StrictSpecializationOf<TypeList> T>
constexpr size_t writableContainersCount() {
  size_t Result = 0;
  compile_time::forEach<T>([&Result]<typename A, size_t I>() {
    if constexpr (not std::is_const_v<A>)
      Result += 1;
  });
  return Result;
}

template<StrictSpecializationOf<TypeList> T>
constexpr size_t writableContainerIndex() {
  int Result = -1;
  compile_time::forEach<T>([&Result]<typename A, size_t I>() {
    if constexpr (not std::is_const_v<A>) {
      Result = I;
    }
  });
  return Result;
}

} // namespace detail

template<typename T>
using SOPTraits = detail::SingleOutputPipeTraits<T, decltype(&T::run)>;

template<typename T, typename C>
class SingleOutputPipeBase {
public:
  static constexpr llvm::StringRef Name = T::Name;
  static constexpr std::array
    ArgumentsDocumentation = C::ArgumentsDocumentation;
  using ContainerTypes = C::ContainerTypes;

  const std::string StaticConfiguration;

  SingleOutputPipeBase(llvm::StringRef Configuration) :
    StaticConfiguration(Configuration.str()) {}

protected:
  static constexpr size_t ContainersSize = std::tuple_size_v<ContainerTypes>;
  static_assert(detail::writableContainersCount<ContainerTypes>() == 1);
  static constexpr size_t
    OutputContainerIndex = detail::writableContainerIndex<ContainerTypes>();
  using OutputContainerType = std::tuple_element_t<OutputContainerIndex,
                                                   ContainerTypes>;
};

template<typename T, typename Base = SingleOutputPipeBase<T, SOPTraits<T>>>
class SingleOutputPipe : public Base {
public:
  template<typename... Args>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    revng_assert(Outgoing[this->OutputContainerIndex].size() == 1);

    detail::ObjectDependeciesHelper ODH(Model, Outgoing, this->ContainersSize);
    T::run(Model, this->StaticConfiguration, Configuration, Containers...);
    ODH.commitUniqueTarget(this->OutputContainerIndex);
    return ODH.takeDependencies();
  }
};

template<typename T>
struct WrappedPipeTraits {
private:
  using ArgumentDocumentation = revng::pypeline::PipeArgumentDocumentation;
  using Arguments = T::Arguments;
  static constexpr size_t Size = std::tuple_size_v<Arguments>;

  template<typename... Args>
  static TypeList<typename Args::Type...> toContainerTypes(TypeList<Args...> *);

  static constexpr std::array<ArgumentDocumentation, Size> documentation() {
    std::array<ArgumentDocumentation, Size> Result;
    compile_time::forEach<Arguments>([&Result]<typename A, size_t I>() {
      Result[I] = ArgumentDocumentation{ .Name = A::Name,
                                         .HelpText = A::HelpText };
    });
    return Result;
  }

public:
  using ContainerTypes = decltype(toContainerTypes(std::declval<
                                                   Arguments *>()));
  static constexpr std::array ArgumentsDocumentation = documentation();
};

template<typename T,
         typename Base = SingleOutputPipeBase<T, WrappedPipeTraits<T>>>
class FunctionPipe : public Base {
private:
  static_assert(Base::OutputContainerType::Kind == Kinds::Function);

public:
  template<typename... Args>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    detail::ObjectDependeciesHelper ODH(Model, Outgoing, this->ContainersSize);
    T Instance(Model, this->StaticConfiguration, Configuration, Containers...);
    const model::Binary &Binary = *Model.get().get();

    for (const ObjectID *Object : Outgoing[this->OutputContainerIndex]) {
      ODH.pushReadFields();

      const MetaAddress &Entry = std::get<MetaAddress>(Object->key());
      // WIP: This adds a read path, is this acceptable?
      const model::Function &Function = Binary.Functions().at(Entry);
      Instance.runOnFunction(Function);
      ODH.commit(*Object, this->OutputContainerIndex);

      ODH.popReadFields();
    }

    return ODH.takeDependencies();
  }
};

template<typename T,
         typename Base = SingleOutputPipeBase<T, WrappedPipeTraits<T>>>
class TypeDefinitionPipe : public Base {
private:
  static_assert(Base::OutputContainerType::Kind == Kinds::TypeDefinition);

public:
  template<typename... Args>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    detail::ObjectDependeciesHelper ODH(Model, Outgoing, this->ContainersSize);
    T Instance(Model, this->StaticConfiguration, Configuration, Containers...);
    const model::Binary &Binary = *Model.get().get();

    for (const ObjectID *Object : Outgoing[this->OutputContainerIndex]) {
      ODH.pushReadFields();

      using TDK = model::TypeDefinition::Key;
      const TDK &Entry = std::get<TDK>(Object->key());
      // WIP: This adds a read path, is this acceptable?
      const UpcastablePointer<model::TypeDefinition>
        &TypeDefinition = Binary.TypeDefinitions().at(Entry);
      Instance.runOnTypeDefinition(TypeDefinition);
      ODH.commit(*Object, this->OutputContainerIndex);

      ODH.popReadFields();
    }
    return ODH.takeDependencies();
  }
};

namespace detail {

template<typename... Args>
inline std::tuple<Args &...> tupleWithRef(TypeList<Args...> *);

template<StrictSpecializationOf<TypeList> T>
using TupleWithRef = decltype(tupleWithRef(std::declval<T *>()));

} // namespace detail

template<typename T,
         typename Base = SingleOutputPipeBase<T, WrappedPipeTraits<T>>>
class LLVMFunctionPassPipe : public Base {
private:
  using LLVMRootContainer = revng::pypeline::LLVMRootContainer;
  static_assert(std::is_same_v<typename Base::OutputContainerType,
                               LLVMRootContainer>);
  using ContainerTypesRef = detail::TupleWithRef<typename Base::ContainerTypes>;

  class Pass : public llvm::ModulePass {
  private:
    detail::ObjectDependeciesHelper &ODH;
    const revng::pypeline::Request &Outgoing;

    const Model &Model;
    llvm::StringRef StaticConfiguration;
    llvm::StringRef Configuration;
    ContainerTypesRef &Containers;

  public:
    static inline char ID = 0;

  public:
    Pass(detail::ObjectDependeciesHelper &ODH,
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
      compile_time::forEach<typename T::Analyses>([&AU]<typename A,
                                                        size_t I>() {
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
      T Instance = Maker(std::make_index_sequence<Base::ContainersSize>());

      const model::Binary &Binary = *Model.get().get();
      for (const ObjectID *Object : Outgoing[Base::OutputContainerIndex]) {
        ODH.pushReadFields();

        const MetaAddress &Entry = std::get<MetaAddress>(Object->key());
        const model::Function &Function = Binary.Functions().at(Entry);
        auto Iter = AddressToFunction.find(Function.Entry());
        revng_assert(Iter != AddressToFunction.end());
        Instance.runOnFunction(Function, *Iter->second);

        ODH.commit(*Object, Base::OutputContainerIndex);
        ODH.popReadFields();
      }

      return true;
    }
  };

private:
  static inline llvm::RegisterPass<Pass> X{ "", "", true, true };

public:
  template<typename... Args>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    detail::ObjectDependeciesHelper ODH(Model, Outgoing, this->ContainersSize);

    llvm::legacy::PassManager Manager;
    compile_time::forEach<typename T::Analyses>([&Manager]<typename A,
                                                           size_t I>() {
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
