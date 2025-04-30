#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/UpcastablePointer.h"
#include "revng/TupleTree/TupleLikeTraits.h"

template<typename O, size_t I = 0>
void initializeOwningPointer(llvm::StringRef Kind,
                             llvm::yaml::IO &TheIO,
                             O &Obj) {
  using concrete_types = concrete_types_traits_t<typename O::element_type>;

  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = typename std::tuple_element_t<I, concrete_types>;
    if (llvm::StringRef(TupleLikeTraits<type>::Name) == Kind) {
      Obj.reset(new type);
    } else {
      initializeOwningPointer<O, I + 1>(Kind, TheIO, Obj);
    }
  } else {
    TheIO.setError("No concrete type for Kind " + Kind.str());
  }
}

template<typename O, size_t I = 0>
void dispatchMappingTraits(llvm::yaml::IO &TheIO, O &Obj) {
  using concrete_types = concrete_types_traits_t<typename O::element_type>;

  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = typename std::tuple_element_t<I, concrete_types>;
    if (type *Upcast = llvm::dyn_cast<type>(Obj.get()))
      llvm::yaml::MappingTraits<type>::mapping(TheIO, *Upcast);
    else
      dispatchMappingTraits<O, I + 1>(TheIO, Obj);
  } else {
    TheIO.setError("Upcastable pointer could not be upcast");
  }
}

template<UpcastablePointerLike T>
struct PolymorphicMappingTraits {
  static void mapping(llvm::yaml::IO &TheIO, T &Obj) {
    // Skip empty pointers when serializing
    if (TheIO.outputting() && Obj.isEmpty())
      return;

    if (!TheIO.outputting()) {
      std::string Kind;
      TheIO.mapRequired("Kind", Kind);
      initializeOwningPointer(Kind, TheIO, Obj);
    }

    dispatchMappingTraits(TheIO, Obj);

    // If kind is default-initialized, clear the pointer.
    if (!TheIO.outputting())
      if (size_t(Obj->Kind()) == 0)
        Obj.reset();
  }
};
