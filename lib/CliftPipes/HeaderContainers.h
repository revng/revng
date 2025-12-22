#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/StringBufferContainer.h"
#include "revng/Pipes/StringMap.h"

namespace revng::kinds {

inline pipeline::SingleElementKind
  TypeAndGlobalHeader("type-and-global-header",
                      Binary,
                      revng::ranks::Binary,
                      fat(revng::ranks::TypeDefinition,
                          revng::ranks::StructField,
                          revng::ranks::UnionField,
                          revng::ranks::EnumEntry,
                          revng::ranks::DynamicFunction,
                          revng::ranks::Segment,
                          revng::ranks::ArtificialStruct,
                          revng::ranks::OpaqueType),
                      { &Decompiled });

} // namespace revng::kinds

namespace detail {

inline constexpr char TypeAndGlobalHeaderName[] = "type-and-global-header";

inline constexpr char HeaderMIMEType[] = "text/x.h+ptml";
inline constexpr char HeaderSuffix[] = ".h";

template<auto... Values>
using SBF = revng::pipes::StringBufferContainer<Values...>;

template<typename T>
using RegisterDCC = pipeline::RegisterDefaultConstructibleContainer<T>;

} // namespace detail

using TypeAndGlobalHeaderContainer = detail::SBF<
  &revng::kinds::TypeAndGlobalHeader,
  detail::TypeAndGlobalHeaderName,
  detail::HeaderMIMEType,
  detail::HeaderSuffix>;
inline detail::RegisterDCC<TypeAndGlobalHeaderContainer> RegisteredMHC;
