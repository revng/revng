#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/ConstexprString.h"
#include "revng/ADT/STLExtras.h"

namespace ptml {

// Attribute in this context is a revng-specific macro that gets unrolled into
// an `__attribute__(($something))` for the compiler.
struct Attribute {
  std::string_view Macro;
  std::string_view Value;
  bool IsReal = false;
};

// Annotation is an attribute that can also encode a value
// (for example an abi name).
struct Annotation {
  std::string_view Macro;
  std::string_view Prefix;
};

struct AttributeRegistry {
  static constexpr std::array<Attribute, 3> StaticAttributes{
    Attribute{ .Macro = "_PACKED", .Value = "packed", .IsReal = true },
    Attribute{ .Macro = "_STACK", .Value = "stack" },
    Attribute{ .Macro = "_CAN_CONTAIN_CODE", .Value = "can_contain_code" }
  };
  static constexpr std::array<Annotation, 5> StaticAnnotations{
    Annotation{ .Macro = "_REG", .Prefix = "reg:" },
    Annotation{ .Macro = "_ABI", .Prefix = "abi:" },
    Annotation{ .Macro = "_STARTS_AT", .Prefix = "field_start_offset:" },
    Annotation{ .Macro = "_SIZE", .Prefix = "struct_size:" },
    Annotation{ .Macro = "_ENUM_UNDERLYING", .Prefix = "enum_underlying_type:" }
  };

  // TODO: add dynamic containers if the need ever arises.

private:
  template<ConstexprString Macro>
  static consteval std::optional<Attribute> getAttributeImpl() {
    auto Result = std::ranges::find_if(StaticAttributes, [](auto &&A) {
      return *Macro == A.Macro;
    });
    if (Result == StaticAttributes.end())
      return std::nullopt;
    return *Result;
  }

  template<ConstexprString Macro>
  static consteval std::optional<Annotation> getAnnotationImpl() {
    auto Result = std::ranges::find_if(StaticAnnotations, [](auto &&A) {
      return *Macro == A.Macro;
    });
    if (Result == StaticAnnotations.end())
      return std::nullopt;
    return *Result;
  }

public:
  template<ConstexprString Macro>
  static std::string getAttribute() {
    constexpr std::optional Attribute = getAttributeImpl<Macro>();
    if constexpr (Attribute) {
      return std::string(Attribute->Macro);
    } else {
      static_assert(value_always_false_v<Macro>, "Unknown attribute.");
    }
  }
  template<ConstexprString Macro>
  static std::string getAnnotation(std::string_view Value) {
    constexpr std::optional Annotation = getAnnotationImpl<Macro>();
    if constexpr (Annotation) {
      return std::string(Annotation->Macro) + "(" + std::string(Value) + ")";
    } else {
      static_assert(value_always_false_v<Macro>, "Unknown annotation.");
    }
  }
  template<ConstexprString Macro>
  static std::string getAnnotation(uint64_t Value) {
    return getAnnotation<Macro>(std::to_string(Value));
  }

  template<ConstexprString Macro>
  static consteval std::string_view getPrefix() {
    constexpr std::optional Annotation = getAnnotationImpl<Macro>();
    if constexpr (Annotation) {
      return Annotation->Prefix;
    } else {
      constexpr std::optional Attribute = getAttributeImpl<Macro>();
      if constexpr (Attribute) {
        return Attribute->Value;
      } else {
        static_assert(value_always_false_v<Macro>,
                      "Unknown attribute or annotation.");
      }
    }
  }

  // TODO: Generate `attributes.h` header instead of providing a static one,
  //       which would enable users to Ctrl + click these macros.
  static std::string getDefinitions() {
    std::string Result;
    for (Attribute const &Attribute : StaticAttributes) {
      Result += "#define " + std::string(Attribute.Macro) + " __attribute__((";
      if (Attribute.IsReal)
        Result += std::string(Attribute.Value);
      else
        Result += "annotate(\"" + std::string(Attribute.Value) + "\")";
      Result += "))\n";
    }

    for (Annotation const &Annotation : StaticAnnotations) {
      Result += "#define " + std::string(Annotation.Macro)
                + "(value) __attribute__((annotate(\""
                + std::string(Annotation.Prefix) + "value\")))\n";
    }

    return Result;
  }

public:
  constexpr bool isMacro(llvm::StringRef String) const {
    auto Comparator = [String](const auto &A) {
      return std::string_view(String) == A.Macro;
    };

    return revng::any_of(StaticAttributes, Comparator)
           || revng::any_of(StaticAnnotations, Comparator);
  }
};
inline constexpr AttributeRegistry Attributes;

} // namespace ptml
