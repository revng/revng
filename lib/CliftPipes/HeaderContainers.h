#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/StringBufferContainer.h"

// TODO: consider moving to all the other kinds
namespace revng::kinds {

inline pipeline::SingleElementKind
  AttributeHeader("attribute-header",
                  Binary,
                  revng::ranks::Binary,
                  fat(revng::ranks::PrimitiveType
                      /* TODO: add location ranks */),
                  {});

inline pipeline::SingleElementKind
  PrimitiveHeader("primitive-header",
                  Binary,
                  ranks::Binary,
                  fat(revng::ranks::PrimitiveType
                      /* TODO: add others */),
                  {});

inline pipeline::SingleElementKind
  NewHelperHeader("new-helper-header",
                  Binary,
                  revng::ranks::Binary,
                  fat(/* TODO: add location ranks */),
                  {});

inline pipeline::SingleElementKind
  NewModelHeader("new-model-header",
                 Binary,
                 revng::ranks::Binary,
                 fat(revng::ranks::TypeDefinition,
                     revng::ranks::StructField,
                     revng::ranks::UnionField,
                     revng::ranks::EnumEntry,
                     revng::ranks::DynamicFunction,
                     revng::ranks::Segment,
                     revng::ranks::ArtificialStruct),
                 { &Decompiled });

} // namespace revng::kinds

namespace detail {

inline constexpr char AttributeHeaderName[] = "attribute-header";
inline constexpr char PrimitiveHeaderName[] = "primitive-header";
inline constexpr char HelperHeaderName[] = "new-helper-header";
inline constexpr char ModelHeaderName[] = "new-model-header";

inline constexpr char HeaderMIMEType[] = "text/x.h+ptml";
inline constexpr char HeaderSuffix[] = ".h";

template<auto... Values>
using SBF = revng::pipes::StringBufferContainer<Values...>;

template<typename T>
using RegisterDCC = pipeline::RegisterDefaultConstructibleContainer<T>;

} // namespace detail

using AttributeHeaderContainer = detail::SBF<&revng::kinds::AttributeHeader,
                                             detail::AttributeHeaderName,
                                             detail::HeaderMIMEType,
                                             detail::HeaderSuffix>;
inline detail::RegisterDCC<AttributeHeaderContainer> RegisteredAHC;

using PrimitiveHeaderContainer = detail::SBF<&revng::kinds::PrimitiveHeader,
                                             detail::PrimitiveHeaderName,
                                             detail::HeaderMIMEType,
                                             detail::HeaderSuffix>;
inline detail::RegisterDCC<PrimitiveHeaderContainer> RegisteredPHC;

using HelperHeaderContainer = detail::SBF<&revng::kinds::NewHelperHeader,
                                          detail::HelperHeaderName,
                                          detail::HeaderMIMEType,
                                          detail::HeaderSuffix>;
inline detail::RegisterDCC<HelperHeaderContainer> RegisteredHHC;

using ModelHeaderContainer = detail::SBF<&revng::kinds::NewModelHeader,
                                         detail::ModelHeaderName,
                                         detail::HeaderMIMEType,
                                         detail::HeaderSuffix>;
inline detail::RegisterDCC<ModelHeaderContainer> RegisteredMHC;
