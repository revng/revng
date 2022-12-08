#pragma once

/*#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#*/
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file is autogenerated! Do not edit it directly

#pragma once

#include <array>

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/TupleTree/TupleLikeTraits.h"
#include "revng/TupleTree/TupleTree.h"

using namespace std::string_view_literals;

/*# --- TupleLikeTraits --- -#*/
template <> struct TupleLikeTraits</*=- struct | user_fullname =*/> {
  static constexpr const llvm::StringRef Name = "/*=- struct.name =*/";
  static constexpr const llvm::StringRef FullName = "/*=- struct | user_fullname =*/";
  using tuple = std::tuple<
    /**- for field in struct.all_fields -**/
    decltype(/*=- struct | user_fullname =*/::/*=- field.name =*/)/** if not loop.last **/, /** endif -**/
    /**- endfor **/>;

  static constexpr std::array<llvm::StringRef, std::tuple_size_v<tuple>> FieldNames = {
    /**- for field in struct.all_fields -**/
    "/*=- field.name =*/"sv,
    /**- endfor **/
  };

  enum class Fields {
    /**- for field in struct.all_fields -**/
    /*=- field.name =*/ = /*=- loop.index0 =*/,
    /**- endfor **/
  };
};

namespace /*= struct.namespace =*/ {
template <int I> auto &get(/*= struct.name =*/ &&x) {
  if constexpr (false)
    return __null;
  /**- for field in struct.all_fields **/
  else if constexpr (I == /*= loop.index0 =*/)
    return x./*= field.name =*/;
  /**- endfor **/
}

template <int I> const auto &get(const /*= struct.name =*/ &x) {
  if constexpr (false)
    return __null;
  /**- for field in struct.all_fields **/
  else if constexpr (I == /*= loop.index0 =*/)
    return x./*= field.name =*/;
  /**- endfor **/
}

template <int I> auto &get(/*= struct.name =*/ &x) {
  if constexpr (false)
    return __null;
  /**- for field in struct.all_fields **/
  else if constexpr (I == /*= loop.index0 =*/)
    return x./*= field.name =*/;
  /**- endfor **/
}
}
/*# --- End TupleLikeTraits --- -#*/

template<>
struct llvm::yaml::MappingTraits</*= struct | user_fullname =*/>
  : public TupleLikeMappingTraits</*= struct | user_fullname =*/
      /**- for field in struct.all_optional_fields -**/
      , TupleLikeTraits</*= struct | user_fullname =*/>::Fields::/*= field.name =*/
      /** endfor -**/
    > {};

/** if struct._key **/
template<>
struct llvm::yaml::ScalarTraits</*= struct | user_fullname =*/::Key>
  : public CompositeScalar</*= struct | user_fullname =*/::Key, '-'> {};
/** endif **//*# --- End YAML traits --- #*/

/*# --- KeyedObjectTraits implementation --- #*/
/** if struct._key **/
/** if struct.keytype == "simple" **/
template<>
struct KeyedObjectTraits</*= struct | user_fullname =*/> {
  static /*= struct.key_fields[0] | field_type =*/ key(const /*= struct | user_fullname =*/ &Obj) { return Obj./*= struct.key_fields[0].name =*/; }
  static /*= struct | user_fullname =*/ fromKey(const /*= struct.key_fields[0] | field_type =*/ &Key) {
    return /*= struct | user_fullname =*/(Key);
  }
};
/** elif struct.keytype == "composite" **/
template<>
struct KeyedObjectTraits</*= struct | user_fullname =*/> {
  using Key = /*= struct | fullname =*/::Key;
  static Key key(const /*= struct | user_fullname =*/ &Obj) {
    return {
      /** for key_field in struct.key_fields -**/
      Obj./*= key_field.name =*/
      /**- if not loop.last **/,
      /** endif **/
      /**- endfor **/
    };
  }

  static /*= struct | user_fullname =*/ fromKey(const Key &K) {
    return std::make_from_tuple</*= struct | user_fullname =*/>(K);
    /*#
    return /*= struct | user_fullname =*/{
      /**- for key_field in struct.key_fields **/
      std::get</*= loop.index0 =*/>(K)/** if not loop.last **/, /** endif **/
      /**- endfor **/
    };
    #*/
  }
};
/** endif **/
/** endif **//*# --- End KeyedObjectTraits implementation --- #*/

/*# --- UpcastablePointer stuff --- #*/
/** if upcastable **/
/// \brief Make UpcastablePointer yaml-serializable polymorphically
template<>
struct llvm::yaml::MappingTraits<UpcastablePointer</*= struct | user_fullname =*/>>
  : public PolymorphicMappingTraits<UpcastablePointer</*= struct | user_fullname =*/>> {};

template<>
struct KeyedObjectTraits<UpcastablePointer</*= struct | user_fullname =*/>> {
  using Key = /*= struct | user_fullname =*/::Key;
  static Key key(const UpcastablePointer</*= struct | user_fullname =*/> &Obj);
  static UpcastablePointer</*= struct | user_fullname =*/> fromKey(const Key &K);
};
/** endif **//*# End UpcastablePointer stuff #*/

static_assert(validateTupleTree</*= struct | user_fullname =*/>(IsYamlizable),
              "/*= struct | user_fullname =*/ must be YAMLizable");

LLVM_YAML_IS_SEQUENCE_VECTOR(/*= struct | user_fullname =*/)
static_assert(Yamlizable<std::vector</*= struct | user_fullname =*/>>,
              "/*= struct | user_fullname =*/ must be YAMLizable");

/** if root_type == struct.name **/
namespace /*= namespace =*/ {

using AllTypes = std::variant<
/** for name in all_types|sort **/
/*= name =*//** if not loop.last **/,/** endif **/
/**- endfor **/
>;

class ConstVisitorBase {
public:
  virtual ~ConstVisitorBase() = default;

/** for name in all_types|sort **/
  virtual void operator()(const /*= name =*/ &Argument) const = 0;
/**- endfor **/
};

template<typename L>
class ConstVisitor : public ConstVisitorBase {
private:
  L *Callable;

public:
  ConstVisitor(L &Callable) : Callable(&Callable) {}
  ~ConstVisitor() override = default;

public:
/** for name in all_types|sort **/
  void operator()(const /*= name =*/ &Argument) const override {
    (*Callable)(Argument);
  }
/**- endfor **/
};

class VisitorBase {
public:
  virtual ~VisitorBase() = default;

/** for name in all_types|sort **/
  virtual void operator()(/*= name =*/ &Argument) const = 0;
/**- endfor **/
};

template<typename L>
class Visitor : public VisitorBase {
private:
  L *Callable;

public:
  Visitor(L &Callable) : Callable(&Callable) {}
  ~Visitor() override = default;

public:
/** for name in all_types|sort **/
  void operator()(/*= name =*/ &Argument) const override {
    (*Callable)(Argument);
  }
/**- endfor **/
};

} // namespace /*= ns =*/

template<>
struct TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/> {
  using ConstVisitorBase = /*= namespace =*/::ConstVisitorBase;

  template<typename L>
  using ConstVisitor = /*= namespace =*/::ConstVisitor<L>;

  using VisitorBase = /*= namespace =*/::VisitorBase;

  template<typename L>
  using Visitor = /*= namespace =*/::Visitor<L>;
};

template<>
struct TupleTreeEntries</*= struct | user_fullname =*/> {
  using Types = /*= namespace =*/::AllTypes;
};

extern template void
TupleTree</*= base_namespace =*/::/*= root_type =*/>::visitImpl(typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::ConstVisitorBase &Pre,
                                    typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::ConstVisitorBase &Post) const;

extern template
void TupleTree</*= base_namespace =*/::/*= root_type =*/>::visitImpl(typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::VisitorBase &Pre,
                                         typename TupleTreeVisitor</*= base_namespace =*/::/*= root_type =*/>::VisitorBase &Post);

extern template
void llvm::yaml::yamlize(llvm::yaml::IO &io, /*= base_namespace =*/::/*= root_type =*/ &Val, bool, llvm::yaml::EmptyContext &Ctx);

extern template
void llvm::yaml::yamlize(llvm::yaml::IO &io, TupleTreeDiff</*= base_namespace =*/::/*= root_type =*/> &Val, bool, llvm::yaml::EmptyContext &Ctx);

extern template
std::optional<TupleTreePath> stringAsPath</*= base_namespace =*/::/*= root_type =*/>(llvm::StringRef Path);

/** endif **/

