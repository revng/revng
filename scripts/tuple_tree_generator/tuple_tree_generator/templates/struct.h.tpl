#pragma once

/*#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#*/
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file is autogenerated! Do not edit it directly

#include <compare>

#include "revng/ADT/TrackingContainer.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/TupleTree/TupleTreeReference.h"
#include "revng/Support/Assert.h"
/**- if emit_tracking **/
#include "revng/Support/AccessTracker.h"
/**- endif **/

void fieldAccessed(llvm::StringRef FieldName, llvm::StringRef StructName);

/** for header in includes **/
#include "/*= user_include_path =*//*= header =*/"
/**- endfor **/

/** if emit_tracking **/
namespace revng {
struct TrackingImpl;
}

struct ReadFields;
/**- endif **/

/*= struct.doc | docstring -=*/
struct /*= struct | fullname =*/
  /**- if struct.inherits **/ : public /*= struct.inherits | user_fullname =*/ /** endif -**/
{
  friend struct revng::TrackingImpl;
  /** if emit_tracking -**/
  inline static constexpr bool HasTracking = true;
  /**- else **/
  inline static constexpr bool HasTracking = false;
  /**- endif **/

  /** if struct.inherits **/
  static constexpr const /*= struct.inherits.name =*/Kind::Values AssociatedKind = /*= struct.inherits.name =*/Kind::/*= struct.name =*/;
  using BaseClass = /*= struct.inherits | user_fullname =*/;
  /**- else **//** if struct.abstract **/
  static constexpr const /*= struct.name =*/Kind::Values AssociatedKind = /*= struct.name =*/Kind::Invalid;
  using BaseClass = /*= struct | user_fullname =*/;
  /**- else **/
  using BaseClass = void;
  /**- endif **//** endif **/

private:
  //
  // Member list
  //
  /**- for field in struct.fields **/
  /*= field | field_type =*/ The/*= field.name =*/ = /*= field | field_type =*/{};
  /**- endfor **/

  /** for field in struct.fields **/
  static_assert(Yamlizable</*= field | field_type =*/>);
  /**- endfor **/

  //
  // Tracking helpers
  //
  /**- if emit_tracking **/
  /**- for field in struct.fields **/
  mutable revng::AccessTracker /*= field.name =*/Tracker = revng::AccessTracker(false);
  /**- endfor **/
  /**- endif **/
  /**- if struct.name == root_type **/
public:
  static constexpr uint64_t SchemaVersion = /*=version=*/;
  /**- endif **/

public:
  //
  // Member accessors
  //
  /**- for field in struct.fields **/

  /*= field.doc | docstring -=*/
  const /*= field | field_type =*/ & /*= field.name =*/() const {
    /**- if emit_tracking **/
    /**- if not field in struct.key_fields **/
    /**- if field.resolved_type.is_scalar **/
#ifdef TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG
    fieldAccessed("/*= field.name =*/" , "/*= struct | fullname =*/");
#endif
    /*= field.name =*/Tracker.access();
    /** endif -**/

    /** endif -**/
    /** endif -**/

    return The/*= field.name =*/;
  }
  /**- endfor **/

  /**- for field in struct.fields **/

  /*= field.doc | docstring -=*/
  /*= field | field_type =*/ & /*= field.name =*/() {
    return The/*= field.name =*/;
  }
  /**- endfor **/

  /** for field in struct.fields **/
  using TypeOf/*= field.name =*/ = /*= field | field_type =*/;
  /**- endfor **/

  /// Default constructor
  /*= struct.name =*/() :
    /**- if struct.inherits **//*= struct.inherits.name =*/()/** endif **/
    /**- for field in struct.fields **/
    /**- if not loop.first or struct.inherits **/, /** endif **/The/*= field.name =*/(
      /**- if struct.name == root_type and field.name == "Version" -**/
        SchemaVersion
      /**- endif -**/
    )
    /**- endfor **/ {
      /**- if struct.inherits -**/
      Kind() = AssociatedKind;
      /**- endif -**/
    }

  /** if struct.abstract **/
  protected:
    /// Copy and move constructors
    /*= struct.name =*/(const /*= struct.name =*/ &Another) = default;
    /*= struct.name =*/(/*= struct.name =*/ &&Another) = default;
    /*= struct.name =*/ &operator=(const /*= struct.name =*/ &Another) = default;
    /*= struct.name =*/ &operator=(/*= struct.name =*/ &&Another) = default;
  public:
  /** else **/
    /// Copy and move constructors
    /*= struct.name =*/(const /*= struct.name =*/ &Another) = default;
    /*= struct.name =*/(/*= struct.name =*/ &&Another) = default;
    /*= struct.name =*/ &operator=(const /*= struct.name =*/ &Another) = default;
    /*= struct.name =*/ &operator=(/*= struct.name =*/ &&Another) = default;
  /** endif **/

  /**- if struct.key_fields **/
  /// Key constructor
  /*= struct.name =*/(
    /**- for field in struct.key_fields **/
    /*=- field | field_type =*/ /*= field.name =*//** if not loop.last **/, /** endif **/
    /**- endfor **/
  ) :
    /**- if struct.inherits **/
    /*=- struct.inherits.name =*/(
      /**- for field in struct.key_fields **/
      /*=- field.name =*//** if not loop.last **/, /** endif **/
      /**- endfor **/
    )
    /**- else **/
    /**- for field in struct.key_fields **/
    The/*=- field.name =*/(/*= field.name =*/)/** if not loop.last **/, /** endif **/
    /**- endfor **/
    /**- endif **/ {
      /**- if struct.inherits and not 'Kind' in struct.key_fields | map(attribute='name') -**/
      Kind() = AssociatedKind;
      /**- endif -**/
    }
  /** endif **/

  /** if struct.emit_full_constructor **/
  /// Full constructor
  /*= struct.name =*/(
    /*#- Inherited fields #*/
    /**- for field in struct.inherits.fields **/
    /**- if field.name != 'Kind' **/
    /*=- field | field_type =*/ /*= field.name =*/
    /** if (struct.fields | length > 0) or (not loop.last) **/, /** endif **/
    /**- endif **/
    /**- endfor **/

    /*#- Own fields #*/
    /*#- Generate for all fields except the implicitly-defined "Version" field of the root type. -#*/
    /**- for field in (struct.fields if struct.name != root_type else struct.fields | rejectattr("name", "equalto", "Version")) | list **/
    /*=- field | field_type =*/ /*= field.name =*/
    /**- if not loop.last **/, /** endif -**/
    /**- endfor **/
  ) :
    /*#- Invoke base class constructor #*/
    /**- if struct.inherits **/
    /*= struct.inherits.name =*/(
      /**- for field in struct.inherits.fields **/
      /**- if field.name != 'Kind' **/
      /*=- field.name =*/
      /**- if not loop.last **/, /** endif -**/
      /**- else **/
      AssociatedKind
      /**- if not loop.last **/, /** endif -**/
      /**- endif **/
      /**- endfor **/
    )
    /** if struct.fields | length > 0 **/, /** endif **/
    /**- endif **/

    /*#- Initialize own fields #*/
    /**- for field in struct.fields **/
    The/*=- field.name =*/(
      /**- if struct.name == root_type and field.name == "Version" -**/
        SchemaVersion
      /**- else -**/
        /*= field.name =*/
      /**- endif -**/
    ) /** if not loop.last **/, /** endif **/
    /**- endfor **/ {}
  /** endif **/

  /** if struct._key **/
  // Key definition for KeyedObjectTraits
  using KeyTuple = std::tuple<
    /**- for key_field in struct.key_fields -**/
    /*= key_field | field_type =*//** if not loop.last **/, /** endif **/
    /**- endfor -**/
  >;
  struct Key : public KeyTuple {
    using KeyTuple::KeyTuple;
  };
  /** endif **/

  // Comparison operators
  /** if not struct.inherits **/
  /** if struct.key_fields -**/
  Key key() const {
    return Key {
      /**- for key_field in struct.key_fields -**/
      /*= key_field.name =*/()/** if not loop.last **/, /** endif **/
      /**- endfor -**/
    };
  }
  bool operator==(const /*= struct.name =*/ &Other) const { return key() == Other.key(); }
  bool operator<(const /*= struct.name =*/ &Other) const { return key() < Other.key(); }
  bool operator>(const /*= struct.name =*/ &Other) const { return key() > Other.key(); }

  /** elif not struct.abstract -**/
  bool operator==(const /*= struct.name =*/ &Other) const {
    /**- for field in struct.fields **/
    if (/*= field.name =*/() != Other./*= field.name =*/())
      return false;
    /**- endfor **/
    return true;
  }
  /** endif **/
  /** endif **/

  /**- if emit_tracking **/
private:
  // Tracking helpers
  template<size_t I>
  revng::AccessTracker& getTracker() const {
    /**- for field in struct.all_fields **/
    if constexpr (I == /*= loop.index0 =*/)
        return /*= field.name =*/Tracker;
    /**- endfor -**/
  }
  /** if upcastable **/
  /**- for child_type in upcastable|sort(attribute="user_fullname") **/
  friend /*= child_type | fullname =*/;
  /**- endfor **/
  /** endif **/

  template<size_t I>
  const auto& untrackedGet() const {
    if constexpr (false)
      return 0;
    /**- for field in struct.all_fields **/
    else if constexpr (I == /*= loop.index0 =*/)
      return The/*= field.name =*/;
    /**- endfor -**/
  }

  /** if struct._key **/
  Key untrackedKey() const {
    return {
      /** for key_field in struct.key_fields -**/
      The/*= key_field.name =*/
      /**- if not loop.last **/,
      /** endif **/
      /**- endfor **/
    };
  }
  /** endif **/

  /** endif -**/
public:
  bool localCompare(const /*= struct | user_fullname =*/ &Other) const;
  void dump(llvm::raw_ostream &Stream) const;
  void dump(std::ostream &Stream) const {
    llvm::raw_os_ostream LLVMStreamAdapter(Stream);
    dump(LLVMStreamAdapter);
  }
  void dump() const debug_function { dump(dbg); }
  void dump(const char *Path) const debug_function;
  std::string toString() const debug_function;

  /**- if struct.abstract **/
public:
  static bool classof(const /*= struct | user_fullname =*/ *P) { return true; }
  /** endif **/

  /**- if struct.inherits **/
public:
  static bool classof(const /*= struct.inherits | user_fullname =*/ *P);
  static bool classof(const /*= struct.inherits.name =*/Kind::Values &Kind) { return Kind == AssociatedKind; }
  /** endif -**/
};

/** if struct._key **/
template<>
struct std::tuple_size</*= struct | fullname =*/::Key>
  : public std::tuple_size</*= struct | fullname =*/::KeyTuple> {};

template<size_t I>
struct std::tuple_element<I, /*= struct | fullname =*/::Key>
  : public std::tuple_element<I, /*= struct | fullname =*/::KeyTuple> {};
/** endif **/

/*# --- UpcastablePointer stuff --- #*/
/** if upcastable **/
/*# Emit both const and non-const specialization of concrete_types_traits #*/
/** for const_qualifier in ["", "const"] **/
template<>
struct concrete_types_traits</*= const_qualifier =*/ /*= struct | user_fullname =*/> {
  using type = std::tuple<
    /**- for child_type in upcastable|sort(attribute="user_fullname") **/
    /*=- const_qualifier =*/ /*= child_type | user_fullname =*//** if not loop.last **/, /** endif **/
    /**- endfor **/
    /**- if not struct.abstract **/, /*=- const_qualifier =*/ /*= struct | user_fullname =*//** endif **/>;
};
/** endfor **/
/** endif **//*# End UpcastablePointer stuff #*/

