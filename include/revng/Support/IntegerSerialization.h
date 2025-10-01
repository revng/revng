#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"

namespace detail {

inline constexpr auto IntegerEndianness = llvm::support::little;
inline constexpr auto Unaligned = llvm::support::unaligned;

template<TupleLike T>
inline constexpr size_t packedTupleSize() {
  size_t Result = 0;
  compile_time::repeat<std::tuple_size_v<T>>([&Result]<size_t I>() {
    using ElementType = std::tuple_element_t<I, T>;
    static_assert(IsIntegralOrEnum<ElementType>);
    Result += sizeof(ElementType);
  });
  return Result;
}

template<typename... T>
struct SerializedIntsSize {};

template<IsIntegralOrEnum... Args>
struct SerializedIntsSize<Args...> {
  static constexpr size_t Size = packedTupleSize<std::tuple<Args...>>();
};

template<TupleLike T>
struct SerializedIntsSize<T> {
  static constexpr size_t Size = packedTupleSize<T>();
};

template<IsIntegralOrEnum T>
void writeNext(uint8_t *&Ptr, const T &Value) {
  // LLVM 19.1+ has writeNext which automatically bumps the pointer, for now we
  // need to bump it manually after using `write`
  llvm::support::endian::write<T, Unaligned>(Ptr, Value, IntegerEndianness);
  Ptr += sizeof(T);
}

template<IsIntegralOrEnum T>
T readNext(const uint8_t *&Ptr) {
  return llvm::support::endian::readNext<T, Unaligned>(Ptr, IntegerEndianness);
}

} // namespace detail

// This is a minimal (de)serialization library for integers and enums. This can
// convert parameters packs and tuples from and to bytes. The members are stored
// in a packed (without alignment padding) little-endian format, this is
// equivalent of `memcpy`-ing the equivalent `packed` struct on x86.

template<typename... T>
inline constexpr size_t packedSize = detail::SerializedIntsSize<T...>::Size;

template<IsIntegralOrEnum... Args>
inline auto toBytes(const Args &...Values) {
  std::array<uint8_t, packedSize<Args...>> Result;
  uint8_t *Ptr = Result.data();
  (detail::writeNext<Args>(Ptr, Values), ...);
  return Result;
}

template<TupleLike T>
inline auto toBytes(const T &Value) {
  return compile_time::callWithIndexSequence<T>([&Value]<size_t... I>() {
    return toBytes(std::get<I>(Value)...);
  });
}

template<IsIntegralOrEnum... ArgsT>
inline void fromBytes(llvm::ArrayRef<uint8_t> Data, ArgsT &...Args) {
  revng_assert(Data.size() == packedSize<ArgsT...>);
  const uint8_t *Ptr = Data.data();
  ((Args = detail::readNext<ArgsT>(Ptr)), ...);
}

template<TupleLike T>
inline void fromBytes(llvm::ArrayRef<uint8_t> Data, T &Value) {
  compile_time::callWithIndexSequence<T>([&Data, &Value]<size_t... I>() {
    fromBytes(Data, std::get<I>(Value)...);
  });
}
