#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <string_view>

/// Iterates over the lines of a string. If the string is empty, the range is
/// empty. Each element line (except the last one) ends in a newline character.
class LineRange {
public:
  class Sentinel {
  public:
    explicit Sentinel() = default;
  };

  class Iterator {
    const char *LineBegin = nullptr;
    const char *LineEnd = nullptr;
    const char *End = nullptr;

  public:
    using value_type = std::string_view;
    using difference_type = ptrdiff_t;

    Iterator() = default;

    explicit Iterator(std::string_view String) :
      LineBegin(String.data()),
      LineEnd(LineBegin),
      End(LineBegin + String.size()) {
      if (LineBegin != End)
        ++(*this);
    }

    [[nodiscard]] std::string_view operator*() const {
      return std::string_view(LineBegin, LineEnd);
    }

    Iterator &operator++() & {
      const char *LF = std::find(LineEnd, End, '\n');
      LineBegin = LineEnd;
      LineEnd = LF == End ? End : LF + 1;
      return *this;
    }

    [[nodiscard]] Iterator operator++(int) & {
      auto Result = *this;
      ++(*this);
      return Result;
    }

    [[nodiscard]] friend bool operator==(const Iterator &Self,
                                         const Sentinel &) {
      return Self.LineBegin == Self.End;
    }
  };

private:
  Iterator Begin;

public:
  explicit LineRange(std::string_view String) : Begin(String) {}

  [[nodiscard]] Iterator begin() const { return Begin; }
  [[nodiscard]] Sentinel end() const { return Sentinel(); }
};
