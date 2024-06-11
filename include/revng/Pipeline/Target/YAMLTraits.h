#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Target.h"
#include "revng/TupleTree/TupleTreeDiff.h"

template<>
struct llvm::yaml::ScalarTraits<const pipeline::Target> {
  static void
  output(const pipeline::Target &Value, void *, llvm::raw_ostream &Output) {
    Output << Value.serialize();
  }

  static StringRef
  input(llvm::StringRef Scalar, void *, const pipeline::Target &Value) {
    // TODO: in order to support deserialization we need some context
    revng_abort();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};

template<>
struct llvm::yaml::ScalarTraits<pipeline::Target> {
  static void
  output(const pipeline::Target &Value, void *, llvm::raw_ostream &Output) {
    llvm::yaml::ScalarTraits<pipeline::Target>::output(Value, nullptr, Output);
  }

  static StringRef
  input(llvm::StringRef Scalar, void *, pipeline::Target &Value) {
    // TODO: in order to support deserialization we need some context
    revng_abort();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};

static_assert(Yamlizable<const pipeline::Target>);
static_assert(Yamlizable<pipeline::Target>);

template<>
struct llvm::yaml::SequenceTraits<const pipeline::TargetsList> {
  static size_t size(IO &IO, const pipeline::TargetsList &Sequence) {
    return Sequence.size();
  }

  static const pipeline::Target &
  element(IO &, const pipeline::TargetsList &Sequence, size_t Index) {
    // TODO: in order to support deserialization we need some context
    revng_assert(Index < Sequence.size());

    return *(Sequence.begin() + Index);
  }
};

template<>
struct llvm::yaml::SequenceTraits<pipeline::TargetsList> {
  static size_t size(IO &IO, pipeline::TargetsList &Sequence) {
    return llvm::yaml::SequenceTraits<
      const pipeline::TargetsList>::size(IO, Sequence);
  }

  static const pipeline::Target &
  element(IO &IO, const pipeline::TargetsList &Sequence, size_t Index) {
    return llvm::yaml::SequenceTraits<
      const pipeline::TargetsList>::element(IO, Sequence, Index);
  }
};

static_assert(Yamlizable<const pipeline::TargetsList>);
static_assert(Yamlizable<pipeline::TargetsList>);

template<typename T>
concept HasEmpty = requires(T &&V) {
  { V.empty() } -> same_as<bool>;
};

template<Yamlizable T>
struct llvm::yaml::MappingTraits<llvm::StringMap<T>> {
  static void mapping(IO &TheIO, llvm::StringMap<T> &Object) {
    for (const auto &[Key, Value] : Object) {
      if constexpr (HasEmpty<T>) {
        if (not Value.empty())
          TheIO.mapOptional(Key.str().c_str(), Value);
      } else {
        TheIO.mapOptional(Key.str().c_str(), Value);
      }
    }
  }
};

template<Yamlizable T>
struct llvm::yaml::MappingTraits<const llvm::StringMap<T>> {
  static void mapping(IO &TheIO, const llvm::StringMap<T> &Object) {
    for (const auto &[Key, Value] : Object)
      TheIO.mapOptional(Key.str().c_str(), Value);
  }
};

static_assert(Yamlizable<llvm::StringMap<pipeline::TargetsList>>);
static_assert(Yamlizable<const llvm::StringMap<pipeline::TargetsList>>);
static_assert(Yamlizable<const llvm::StringMap<const pipeline::TargetsList>>);

template<>
struct llvm::yaml::MappingTraits<const pipeline::ContainerToTargetsMap> {
  static void mapping(IO &TheIO, const pipeline::ContainerToTargetsMap &Info) {
    llvm::yaml::MappingTraits<
      const llvm::StringMap<pipeline::TargetsList>>::mapping(TheIO,
                                                             Info.Status);
  }
};

template<>
struct llvm::yaml::MappingTraits<pipeline::ContainerToTargetsMap> {
  static void mapping(IO &TheIO, pipeline::ContainerToTargetsMap &Info) {
    llvm::yaml::MappingTraits<
      llvm::StringMap<pipeline::TargetsList>>::mapping(TheIO, Info.Status);
  }
};

static_assert(Yamlizable<pipeline::ContainerToTargetsMap>);
static_assert(Yamlizable<const pipeline::ContainerToTargetsMap>);

static_assert(Yamlizable<pipeline::TargetInStepSet>);
static_assert(Yamlizable<const pipeline::TargetInStepSet>);
