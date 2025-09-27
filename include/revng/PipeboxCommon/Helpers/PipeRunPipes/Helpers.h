#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Model.h"

//
// ObjectDependenciesHelper
//

/// Helper class that handles most of the heavy lifting around generating a
/// `ObjectDependencies` object. This is model-aware and uses the model tracking
/// functionality to automatically populate the `ObjectDependencies` object.
class ObjectDependenciesHelper {
private:
  const model::Binary &Binary;
  const revng::pypeline::Request &Outgoing;
  revng::pypeline::ObjectDependencies Dependencies;
  bool DependenciesTaken = false;

public:
  ObjectDependenciesHelper(const Model &Model,
                           const revng::pypeline::Request &Outgoing,
                           size_t DependenciesSize) :
    Binary(*Model.get().get()),
    Outgoing(Outgoing),
    Dependencies(DependenciesSize) {
    revng::Tracking::clearAndResume(Binary);
  }
  ~ObjectDependenciesHelper() { revng_assert(DependenciesTaken); }

  ObjectDependenciesHelper(const ObjectDependenciesHelper &) = delete;
  ObjectDependenciesHelper &
  operator=(const ObjectDependenciesHelper &) = delete;
  ObjectDependenciesHelper(ObjectDependenciesHelper &&) = delete;
  ObjectDependenciesHelper &operator=(ObjectDependenciesHelper &&) = delete;

  /// Given the ArgumentIndex-th container, add all the read paths obtained by
  /// tracking to the specified ObjectID
  void commit(const ObjectID &Object, size_t ArgumentIndex) {
    revng_assert(ArgumentIndex < Dependencies.size());
    std::vector<std::string> ReadPaths = getReadPaths();
    for (std::string &Path : ReadPaths)
      Dependencies[ArgumentIndex].push_back({ Object, Path });
  }

  /// Given the ArgumentIndex-th container, add all the read paths obtained by
  /// tracking to all the requested Outgoing objects
  void commitAllFor(size_t ArgumentIndex) {
    revng_assert(ArgumentIndex < Dependencies.size());
    std::vector<std::string> ReadPaths = getReadPaths();
    for (const ObjectID *Object : Outgoing[ArgumentIndex])
      for (std::string &Path : ReadPaths)
        Dependencies[ArgumentIndex].push_back({ *Object, Path });
  }

  /// In the special-case that a single ObjectID is requested, add all the read
  /// paths to it as a dependency
  void commitUniqueTarget(size_t ArgumentIndex) {
    revng_assert(ArgumentIndex < Dependencies.size());
    revng_assert(Outgoing[ArgumentIndex].size() == 1);
    commitAllFor(ArgumentIndex);
  }

  /// Add a "layer" to the tracking, the paths read by the `commit*` functions
  /// will only be present until the call to `popReadFields`
  void pushReadFields() { revng::Tracking::push(Binary); }

  /// Remove a "layer" to the tracking, paths read since the last
  /// `pushReadFields` will no longer be present when `commit*` is called
  void popReadFields() { revng::Tracking::pop(Binary); }

  /// Retrieve the ObjectDependencies. This needs to be called exactly once,
  /// otherwise an assertion will be hit.
  [[nodiscard]] revng::pypeline::ObjectDependencies &&takeDependencies() {
    revng_assert(not DependenciesTaken);
    DependenciesTaken = true;
    return std::move(Dependencies);
  }

private:
  /// RAII object that automatically does `pushReadFields` on construction and
  /// `commit`, `popReadFields` on destruction.
  class Committer {
  private:
    ObjectDependenciesHelper &ODH;
    const ObjectID &Object;
    size_t ArgumentIndex;

  public:
    Committer(ObjectDependenciesHelper &ODH,
              const ObjectID &Object,
              size_t ArgumentIndex) :
      ODH(ODH), Object(Object), ArgumentIndex(ArgumentIndex) {
      ODH.pushReadFields();
    }

    ~Committer() {
      ODH.commit(Object, ArgumentIndex);
      ODH.popReadFields();
    }

    Committer(const Committer &) = delete;
    Committer &operator=(const Committer &) = delete;
    Committer(Committer &&) = delete;
    Committer &operator=(Committer &&) = delete;
  };

public:
  [[nodiscard]] Committer getCommitterFor(const ObjectID &Object,
                                          size_t ArgumentIndex) {
    return Committer(*this, Object, ArgumentIndex);
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

//
// Define the two concepts IsSingleObjectPipeRun and IsMultipleObjectsPipeRun
// which are the two categories of Pipe Runs
//

namespace detail {

template<typename T>
concept IsPipeArgument = requires {
  requires IsPipeArgumentDocumentation<T>;
  requires IsContainerArgument<typename T::Type>;
};

template<SpecializationOf<TypeList> List>
inline constexpr bool checkArguments() {
  constexpr bool Size = std::tuple_size_v<List>;
  return compile_time::repeatAnd<Size>([]<size_t I>() {
    return IsPipeArgument<std::tuple_element_t<I, List>>;
  });
}

template<typename T>
concept HasArguments = requires {
  requires StrictSpecializationOf<typename T::Arguments, TypeList>;
  requires checkArguments<typename T::Arguments>();
};

template<typename T>
concept HasStaticRun = std::is_function_v<
  std::remove_pointer_t<decltype(&T::run)>>;

} // namespace detail

template<typename T>
concept IsSingleObjectPipeRun = requires {
  requires HasName<T>;
  requires detail::HasStaticRun<T>;
  requires not detail::HasArguments<T>;
};

template<typename T>
concept IsMultipleObjectsPipeRun = requires {
  requires HasName<T>;
  requires detail::HasArguments<T>;
  requires not detail::HasStaticRun<T>;
};

//
// PipeRunContainerTypes
//

namespace detail {

template<typename T>
struct PipeRunContainerTypesImpl {};

template<IsMultipleObjectsPipeRun T>
struct PipeRunContainerTypesImpl<T> {
private:
  template<typename... Args>
  static TypeList<typename Args::Type...> toContainerTypes(TypeList<Args...>);

public:
  using ContainerTypes = decltype(toContainerTypes(std::declval<
                                                   typename T::Arguments>()));
};

template<typename T>
struct SingleOutputPipeTraits {};

template<typename... Args>
struct SingleOutputPipeTraits<
  void (*)(const Model &, llvm::StringRef, llvm::StringRef, Args &...)> {
  using ContainerTypes = TypeList<Args...>;
};

template<IsSingleObjectPipeRun T>
struct PipeRunContainerTypesImpl<T> {
  using ContainerTypes = SingleOutputPipeTraits<
    decltype(&T::run)>::ContainerTypes;
};

} // namespace detail

/// This using will return a TypeList of container types (including const) used
/// by a Pipe Run class. This works for both SingleObjectPipeRun and
/// MultipleObjectPipeRun classes.
template<typename T>
using PipeRunContainerTypes = detail::PipeRunContainerTypesImpl<
  T>::ContainerTypes;

//
// Concepts and helpers used around hasConstructor
//

namespace detail {

template<typename T, typename TL, size_t... I>
concept HasConstructor = requires {
  { T{ std::declval<std::tuple_element_t<I, TL>>()... } } -> std::same_as<T>;
};

template<typename... Args>
TypeList<Args &...> addReference(TypeList<Args...>);

template<typename T>
using AddReference = decltype(addReference(std::declval<T>()));

} // namespace detail

/// This bool will be true if T can be constructed with values specified in the
/// TypeList TL. The main use is in concept to assert the shape of the
/// constructor.
template<typename T, SpecializationOf<TypeList> TL>
constexpr bool hasConstructor() {
  auto Runner = []<size_t... I>(std::index_sequence<I...>) {
    return detail::HasConstructor<T, TL, I...>;
  };
  return Runner(std::make_index_sequence<std::tuple_size_v<TL>>());
};

/// Helper using that will return a TypeList of ContainerTypes with `&` added
template<typename T>
using ConstructorContainerArguments = detail::AddReference<
  PipeRunContainerTypes<T>>;
