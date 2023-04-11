/*#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#*/
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/TupleTree/VisitsImpl.h"
#include "revng/TupleTree/TupleTreeImpl.h"

/**- for child_type in upcastable **/
#include "/*= generator.user_include_path =*//*= child_type.name =*/.h"
/**- endfor **/

#include "/*= generator.user_include_path =*//*= struct.name =*/.h"

/** if upcastable **/

using Key = /*= struct | user_fullname =*/::Key;

Key KeyedObjectTraits<UpcastablePointer</*= struct | user_fullname =*/>>::key(
  const UpcastablePointer</*= struct | user_fullname =*/> &Obj)
{

  return {
    /**- for key_field in struct.key_fields **/
    Obj->/*= key_field.name =*/()/** if not loop.last **/, /** endif **/
    /**- endfor **/
  };
}

UpcastablePointer</*= struct | user_fullname =*/>
KeyedObjectTraits<UpcastablePointer</*= struct | user_fullname =*/>>::fromKey(
  const Key &K)
{
  using namespace model;
  using ResultType = UpcastablePointer</*= struct | user_fullname =*/>;
  /**- for child_type in upcastable|sort(attribute="user_fullname") **/
  if (/*= child_type | user_fullname =*/::classof(K)) {
    auto *Tmp = new /*= child_type | user_fullname =*/(
      /**- for key_field in child_type.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/);
    return ResultType(Tmp);
  }
  /** if not loop.last **/else /** endif **/
  /**- endfor **/
  /** if not struct.abstract **/
  else if (/*= struct | user_fullname =*/::classof(K)) {
    auto *Tmp = new /*= struct | user_fullname =*/(
      /**- for key_field in struct.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/);
    return ResultType(Tmp);
  }
  /** endif **/
  else {
    return ResultType(nullptr);
  }
}
/** endif **/

bool /*= struct | fullname =*/::localCompare(const /*= struct | user_fullname =*/ &Other) const {
  /**- if struct.abstract **/

  auto *Left = static_cast<const /*= struct | user_fullname =*/ *>(this);
  auto *Right = &Other;
  return upcast(Left, [&Right](const auto &UpcastedL) -> bool {
    return upcast(Right, [&UpcastedL](const auto &UpcastedR) -> bool{
      if constexpr (not std::is_same_v<decltype(UpcastedL), decltype(UpcastedR)>) {
        return false;
      } else {
        return UpcastedL.localCompare(UpcastedR);
      }
    }, false);
  }, false);

  /**- else -**/

  /** for field in struct.all_fields if not field.is_guid and field.__class__.__name__ != "ReferenceStructField" **/

  /**- if field.__class__.__name__ == "SimpleStructField" **/

  /**- if schema.get_definition_for(field.type).__class__.__name__ == "StructDefinition" -**/
  if (not this->/*= field.name =*/().localCompare(Other./*= field.name =*/()))
    return false;
  /**- else -**/
  if (this->/*= field.name =*/() != Other./*= field.name =*/())
    return false;
  /**- endif -**/

  /**- elif field.__class__.__name__ == "SequenceStructField" -**/
  if (this->/*= field.name =*/().size() != Other./*= field.name =*/().size())
    return false;

  /**- if schema.get_definition_for(field.element_type).__class__.__name__ == "StructDefinition" -**/
  for (const auto &[L, R] : llvm::zip(this->/*= field.name =*/(), Other./*= field.name =*/())) {
    /** if field.upcastable **/
    if (not L->localCompare(*R))
      return false;
    /** else **/
    if (not L.localCompare(R))
      return false;
    /** endif **/
  }

  /**- else -**/
  if (this->/*= field.name =*/() != Other./*= field.name =*/())
    return false;
  /**- endif -**/

  /** else **//*= ERROR("unexpected field type") =*//** endif **/

  /** endfor **/

  return true;
  /**- endif -**/
}

/** if struct.name == root_type **/

template void
TupleTree</*= base_namespace =*/::/*= root_type =*/>::visitImpl(typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::ConstVisitorBase &Pre,
                                    typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::ConstVisitorBase &Post) const;

template
void TupleTree</*= base_namespace =*/::/*= root_type =*/>::visitImpl(typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::VisitorBase &Pre,
                                         typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::VisitorBase &Post);

template
void llvm::yaml::yamlize(llvm::yaml::IO &io, /*= base_namespace =*/::/*= root_type =*/ &Val, bool, llvm::yaml::EmptyContext &Ctx);

template
void llvm::yaml::yamlize(llvm::yaml::IO &io, TupleTreeDiff</*= base_namespace =*/::/*= root_type =*/> &Val, bool, llvm::yaml::EmptyContext &Ctx);

template
TupleTreeDiff</*= base_namespace =*/::/*= root_type =*/> diff(const /*= base_namespace =*/::/*= root_type =*/ &LHS, const /*= base_namespace =*/::/*= root_type =*/ &RHS);

template
std::optional<TupleTreePath> stringAsPath</*= base_namespace =*/::/*= root_type =*/>(llvm::StringRef Path);

/** endif **/

/**- for field in struct.fields **/
/**- if struct | is_struct_field **/
static_assert(not (TupleTreeCompatible</*= field | field_type =*/> and KeyedObjectContainerCompatible</*= field | field_type =*/>));
/**- endif **/
/**- endfor **/
