#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <climits>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Debug.h"

namespace yield::ptml {

namespace tags {

inline constexpr auto Div = "div";
inline constexpr auto Span = "span";

} // namespace tags

namespace attributes {

inline constexpr auto scope = "data-scope";
inline constexpr auto token = "data-token";
inline constexpr auto locationDefinition = "data-location-definition";
inline constexpr auto locationReferences = "data-location-references";
inline constexpr auto modelEditPath = "data-model-edit-path";

inline constexpr auto htmlExclusiveMetadata = "data-html-exclusive";

} // namespace attributes

namespace scopes {

inline constexpr auto Annotation = "annotation";
inline constexpr auto Comment = "comment";
inline constexpr auto Error = "error";
inline constexpr auto Indentation = "indentation";
inline constexpr auto Link = "link";

} // namespace scopes

class Tag {
private:
  std::string TheTag;
  std::string Content;
  llvm::StringMap<std::string> Attributes;

public:
  explicit Tag() {}
  explicit Tag(llvm::StringRef TheTag, std::string &&Content) :
    TheTag(TheTag.str()), Content(std::move(Content)) {}
  explicit Tag(llvm::StringRef TheTag, llvm::StringRef Content = "") :
    Tag(TheTag, Content.str()) {}

  struct Scope {
  private:
    const Tag &TheTag;
    llvm::raw_ostream &OS;

  public:
    Scope(const Tag &TheTag, llvm::raw_ostream &OS) : TheTag(TheTag), OS(OS) {
      OS << TheTag.open();
    }

    ~Scope() { OS << TheTag.close(); }
  };

  Tag &add(llvm::StringRef Name, llvm::StringRef Value) {
    Attributes[Name] = Value.str();
    return *this;
  }

  template<::ranges::typed<llvm::StringRef> T>
  Tag &add(llvm::StringRef Name, const T &Values) {
    Attributes[Name] = llvm::join(Values, ",");
    return *this;
  }

  template<typename... T>
  requires(std::is_convertible_v<llvm::StringRef, T> and...)
    Tag &add(llvm::StringRef Name, T &&...Values) {
    Attributes[Name] = llvm::join(",", std::forward<T>(Values)...);
    return *this;
  }

  std::string open() const {
    llvm::SmallString<128> Out;
    Out.append({ "<", TheTag, " " });
    for (auto &Pair : Attributes)
      Out.append({ Pair.first(), "=\"", Pair.second, "\" " });
    Out.pop_back();
    Out += '>';
    return Out.str().str();
  }

  std::string close() const { return "</" + TheTag + ">"; }

  std::string serialize() const { return open() + Content + close(); }

  Scope scope(llvm::raw_ostream &OS) { return Scope(*this, OS); }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << serialize();
  }

  bool verify() const debug_function { return !TheTag.empty(); }
};

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

class PTMLIndentedOstream : public llvm::raw_ostream {
private:
  int IndentSize;
  int IndentDepth;
  bool TrailingNewline;
  raw_ostream &OS;

public:
  explicit PTMLIndentedOstream(llvm::raw_ostream &OS, int IndentSize = 2) :
    IndentSize(IndentSize), IndentDepth(0), TrailingNewline(false), OS(OS) {
    SetUnbuffered();
  }

  void indent() { IndentDepth = std::min(INT_MAX, IndentDepth + 1); }
  void unindent() { IndentDepth = std::max(0, IndentDepth - 1); }

private:
  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override;
  void writeIndent();
};

} // namespace yield::ptml
