#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

/// \brief Utility class template for heterogeneous comparison of pointers with
/// unique pointers.
//
//  This is particularly useful when you have a std::set<std::unique_ptr<T>> or
//  a std::map<std::unique_ptr<T1>, T2>. In general you cannot perform lookups
//  using raw pointers into such containers.
//  Using this class template you can transform the into
//  std::set<std::unique_ptr<T>, HeterogeneousPtrCompare<T>> and
//  std::map<std::unique_ptr<T1>, T2, HeterogeneousPtrCompare<T1>>, which allows
//  lookup using raw pointers.
template<typename T>
struct HeterogeneousPtrCompare {
  using is_transparent = std::true_type;

private:
  struct Helper {
    const T *P;
    Helper() = default;
    ~Helper() = default;
    Helper(const Helper &) = default;
    Helper(Helper &&) = default;
    Helper &operator=(const Helper &) = default;
    Helper &operator=(Helper &&) = default;
    Helper(const T *Ptr) : P(Ptr) {}
    template<typename DeleterT>
    Helper(const std::unique_ptr<T, DeleterT> &Ptr) : Helper(Ptr.get()) {}
  };

public:
  bool operator()(const Helper A, const Helper B) const { return A.P < B.P; }
};
