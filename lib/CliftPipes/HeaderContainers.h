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

inline pipeline::SingleElementKind
  HelperHeader("helper-header", Binary, revng::ranks::Binary, {}, {});

inline TypeKind SingleTypeDefinition("single-type-definition",
                                     TypeAndGlobalHeader,
                                     ranks::TypeDefinition,
                                     {},
                                     {});

} // namespace revng::kinds

namespace detail {

inline constexpr char TypeAndGlobalHeaderName[] = "type-and-global-header";
inline constexpr char HelperHeaderName[] = "helper-header";

inline constexpr char HeaderMIMEType[] = "text/x.h+ptml";
inline constexpr char HeaderSuffix[] = ".h";

inline constexpr char TypeDefinitionName[] = "single-type-definition";
inline constexpr char TypeDefinitionMimeType[] = "text/x.c+tar+gz";
inline constexpr char TypeDefinitionSuffix[] = ".c";

template<auto... Values>
using SBF = revng::pipes::StringBufferContainer<Values...>;

// The real class is used here because aliasing an alias is not allowed.
namespace RPD = revng::pipes::detail;
template<auto... Values>
using TSM = RPD::GenericStringMap<&revng::ranks::TypeDefinition, Values...>;

template<typename T>
using RegisterDCC = pipeline::RegisterDefaultConstructibleContainer<T>;

} // namespace detail

using TypeAndGlobalHeaderContainer = detail::SBF<
  &revng::kinds::TypeAndGlobalHeader,
  detail::TypeAndGlobalHeaderName,
  detail::HeaderMIMEType,
  detail::HeaderSuffix>;
inline detail::RegisterDCC<TypeAndGlobalHeaderContainer> RegisteredMHC;

using HelperHeaderContainer = detail::SBF<&revng::kinds::HelperHeader,
                                          detail::HelperHeaderName,
                                          detail::HeaderMIMEType,
                                          detail::HeaderSuffix>;
inline detail::RegisterDCC<HelperHeaderContainer> RegisteredHHC;

using TypeDefinitionContainer = detail::TSM<&revng::kinds::SingleTypeDefinition,
                                            detail::TypeDefinitionName,
                                            detail::TypeDefinitionMimeType,
                                            detail::TypeDefinitionSuffix>;
inline detail::RegisterDCC<TypeDefinitionContainer> RegisteredTDC;
