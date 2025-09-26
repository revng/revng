/*#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#*/
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/** if root_type **/
#include "/*= user_include_path =*//*= root_type =*/.h"
/** endif **/
#include "revng/TupleTree/VisitsImpl.h"
#include "revng/TupleTree/TupleTreeImpl.h"
#include "revng/TupleTree/TrackingImpl.h"

/**- for child_type in upcastable **/
#include "/*= user_include_path =*//*= child_type.name =*/.h"
/**- endfor **/

#include "/*= user_include_path =*//*= struct.name =*/.h"

/** if upcastable and struct.key_definition._key **/

using /*= struct.name =*/Key = /*= struct | user_fullname =*/::Key;
using U/*= struct.name =*/ = /*= base_namespace =*/::Upcastable/*= struct.name =*/;
using /*= struct.name =*/KOT = KeyedObjectTraits<U/*= struct.name =*/>;

/*= struct.name =*/Key /*= struct.name =*/KOT::key(const U/*= struct.name =*/ &Obj) {
  return {
    /**- for key_field in struct.key_fields **/
    Obj->/*= key_field.name =*/()/** if not loop.last **/, /** endif **/
    /**- endfor **/
  };
}

U/*= struct.name =*/ /*= struct.name =*/KOT::fromKey(const /*= struct.name =*/Key &K) {
  using namespace model;
  /**- for child_type in upcastable|sort(attribute="user_fullname") **/
  if (/*= child_type | user_fullname =*/::classof(std::get</*= struct._key_kind_index =*/>(K))) {
    auto *Tmp = new /*= child_type | user_fullname =*/(
      /**- for key_field in child_type.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/);
    return U/*= struct.name =*/(Tmp);
  }
  /**- if not loop.last **/else /** endif **/
  /**- endfor **/
  /** if not struct.abstract **/
  else if (/*= struct | user_fullname =*/::classof(std::get</*= struct._key_kind_index =*/>(K))) {
    auto *Tmp = new /*= struct | user_fullname =*/(
      /**- for key_field in struct.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/);
    return U/*= struct.name =*/(Tmp);
  }
  /**- endif -**/
  else {
    return U/*= struct.name =*/::empty();
  }
}

/** endif **/

/** if upcastable **/

/*= base_namespace =*/::Upcastable/*= struct.name =*/
/*= base_namespace =*/::copy/*= struct.name =*/(const /*= struct.name =*/ &From) {
  Upcastable/*= struct.name =*/ Result;
  upcast(&From, [&Result]<typename T>(const T &Upcasted) {
    Result = Upcastable/*= struct.name =*/::make<T>(Upcasted);
  });
  return Result;
}

template
bool UpcastablePointer</*= struct | user_fullname =*/>::operator==(const UpcastablePointer &Other) const;

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
  /**- if field.upcastable -**/
  if (this->/*= field.name =*/().isEmpty() || Other./*= field.name =*/().isEmpty()) {
    if (this->/*= field.name =*/() != Other./*= field.name =*/())
      return false;
  } else if (not this->/*= field.name =*/()->localCompare(*Other./*= field.name =*/()))
  /**- else -**/
  if (not this->/*= field.name =*/().localCompare(Other./*= field.name =*/()))
  /**- endif -**/
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

void /*= struct | fullname =*/::dump(llvm::raw_ostream &Stream) const {
  auto *This = static_cast<const /*= struct | user_fullname =*/ *>(this);

  /**- if emit_tracking **/
  DisableTracking Guard(*This);
  /**- endif **/

  /**- if upcastable **/
  upcast(This, [&Stream](auto &Upcasted) { serialize(Stream, Upcasted); });
  /**- else **/
  serialize(Stream, *This);
  /** endif -**/
}

void /*= struct | fullname =*/::dump(const char *Path) const {
  std::error_code EC;
  llvm::raw_fd_stream Stream(Path, EC);
  revng_assert(not EC);
  dump(Stream);
}

std::string /*= struct | fullname =*/::toString() const {
  std::string Buffer;
  llvm::raw_string_ostream StringStream(Buffer);

  dump(StringStream);
  return Buffer;
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

template
std::optional<std::string> pathAsString</*= base_namespace =*/::/*= root_type =*/>(const TupleTreePath &Path);

template
bool TupleTree</*= base_namespace =*/::/*= root_type =*/>::verifyReferences(bool Assert) const;

/** endif **/

/**- if emit_tracking **/

template
ReadFields revng::Tracking::collect(const /*= base_namespace =*/::/*= struct.name =*/ &LHS);

template
void revng::Tracking::clearAndResume(const /*= base_namespace =*/::/*= struct.name =*/ &LHS);

template
void revng::Tracking::push(const /*= base_namespace =*/::/*= struct.name =*/ &LHS);

template
void revng::Tracking::pop(const /*= base_namespace =*/::/*= struct.name =*/ &LHS);

template
void revng::Tracking::stop(const /*= base_namespace =*/::/*= struct.name =*/ &LHS);

/** endif **/

/**- for field in struct.fields **/
/**- if field | is_struct_field **/
static_assert(not (TupleTreeCompatible</*= field | field_type =*/> and KeyedObjectContainerCompatible</*= field | field_type =*/>));
/**- endif **/
/**- endfor **/
