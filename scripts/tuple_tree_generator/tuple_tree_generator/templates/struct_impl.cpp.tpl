/*#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#*/
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/** if upcastable **/
/**- for child_type in upcastable|sort(attribute="user_fullname") **/
#include "/*= generator.user_include_path =*//*= child_type.filename =*/"
/**- endfor **/

#include "/*= generator.user_include_path =*//*= struct.filename =*/"

using Key = /*= struct.user_fullname =*/::Key;

Key KeyedObjectTraits<UpcastablePointer</*= struct.user_fullname =*/>>::key(
  const UpcastablePointer</*= struct.user_fullname =*/> &Obj)
{

  return {
    /**- for key_field in struct.key_fields **/
    Obj->/*= key_field.name =*//** if not loop.last **/, /** endif **/
    /**- endfor **/
  };
}

UpcastablePointer</*= struct.user_fullname =*/>
KeyedObjectTraits<UpcastablePointer</*= struct.user_fullname =*/>>::fromKey(
  const Key &K)
{
  using namespace model;
  using ResultType = UpcastablePointer</*= struct.user_fullname =*/>;
  /**- for child_type in upcastable|sort(attribute="user_fullname") **/
  if (/*= child_type.user_fullname =*/::classof(K)) {
    auto *Tmp = new /*= child_type.user_fullname =*/(
      /**- for key_field in child_type.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/);
    return ResultType(Tmp);
  }
  /** if not loop.last **/else /** endif **/
  /**- endfor **/
  /** if not struct.abstract **/
  else if (/*= struct.user_fullname =*/::classof(K)) {
    auto *Tmp = new /*= struct.user_fullname =*/(
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
