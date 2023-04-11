#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <climits>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include "revng/ADT/Concepts.h"
#include "revng/PTML/Constants.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

namespace ptml {

struct ScopeTag;

class Tag {
private:
  std::string TheTag;
  std::string Content;
  llvm::StringMap<std::string> Attributes;

public:
  Tag() {}
  explicit Tag(llvm::StringRef Tag) : TheTag(Tag.str()) {}
  explicit Tag(llvm::StringRef Tag, std::string &&Content) :
    TheTag(Tag.str()), Content(std::move(Content)) {}

  ScopeTag scope(llvm::raw_ostream &OS, bool Newline = false) const;

  Tag &setContent(const llvm::StringRef Content) {
    this->Content = Content.str();
    return *this;
  }

  Tag &addAttribute(llvm::StringRef Name, llvm::StringRef Value) {
    Attributes[Name] = Value.str();
    return *this;
  }

  template<range_with_value_type<llvm::StringRef> T>
  Tag &addListAttribute(llvm::StringRef Name, const T &Values) {
    for (auto &Value : Values)
      revng_check(!llvm::StringRef(Value).contains(","));
    Attributes[Name] = llvm::join(Values, ",");
    return *this;
  }

  // clang-format off
  template<typename... T>
    requires(std::is_convertible_v<T, llvm::StringRef> and ...)
  Tag &addListAttribute(llvm::StringRef Name, const T &...Value) {
    // clang-format on
    std::initializer_list<llvm::StringRef> Values = { Value... };
    return this->addListAttribute(Name, Values);
  }

  std::string open() const {
    llvm::SmallString<128> Out;
    Out.append({ "<", TheTag });
    for (auto &Pair : Attributes)
      Out.append({ " ", Pair.first(), "=\"", Pair.second, "\"" });
    Out.append({ '>' });
    return Out.str().str();
  }

  std::string close() const { return "</" + TheTag + ">"; }

  std::string serialize() const { return open() + Content + close(); }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << serialize();
  }

  bool verify() const debug_function { return !TheTag.empty(); }
};

inline Tag scopeTag(const llvm::StringRef AttributeName) {
  return Tag(ptml::tags::Div)
    .addAttribute(ptml::attributes::Scope, AttributeName);
}

inline std::string operator+(const Tag &LHS, const llvm::StringRef RHS) {
  return LHS.serialize() + RHS.str();
}

inline std::string operator+(const llvm::StringRef LHS, const Tag &RHS) {
  return LHS.str() + RHS.serialize();
}

inline std::string operator+(const Tag &LHS, const Tag &RHS) {
  return LHS.serialize() + RHS.serialize();
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Tag &TheTag) {
  OS << TheTag.serialize();
  return OS;
}

template<typename T>
inline std::string str(const T &Obj) {
  return getNameFromYAMLScalar(Obj);
};

/// Helper class that allows RAII-style handling of content-less tags, opening
/// them at construction and closing them when the object goes out of scope
/// \code{.cpp}
/// Out << "Foo"
/// {
///        auto Scope = Tag(tags::Span).scope(Out);
///        Out << "Bar"
/// } // Out of scope, </span> will be emitted
/// \endcode
/// the parameter \param NewLine will output a newline after the opening tag
struct ScopeTag {
private:
  llvm::raw_ostream &OS;
  const std::string TagClose;

public:
  ScopeTag(llvm::raw_ostream &OS, const Tag &TheTag, bool Newline) :
    OS(OS), TagClose(TheTag.close()) {
    OS << TheTag.open();
    if (Newline)
      OS << "\n";
  }

  ~ScopeTag() { OS << TagClose; }
};

inline ScopeTag Tag::scope(llvm::raw_ostream &OS, bool Newline) const {
  revng_check(Content.empty());
  return ScopeTag(OS, *this, Newline);
}

} // namespace ptml
