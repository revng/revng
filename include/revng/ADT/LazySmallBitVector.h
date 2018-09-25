#ifndef LAZYSMALLBITVECTOR_H
#define LAZYSMALLBITVECTOR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <climits>
#include <cstdint>
#include <cstring>
extern "C" {
#include <strings.h>
}
#include <limits>

// Boost includes
#include <boost/iterator/iterator_facade.hpp>

// Local libraries includes
#include "revng/Support/Assert.h"

// TODO: implement shrinking

// TODO: implement using __builtin_clz if available
/// \brief Returns the minimum amount of bits required to represent \p Value
template<typename T>
inline unsigned requiredBits(T Value) {
  unsigned Result = 0;

  while (Value != 0) {
    Result++;
    Value = Value >> 1;
  }

  return Result;
}

template<typename T, typename A, typename B>
constexpr bool is_either() {
  return std::is_same<T, A>::value || std::is_same<T, B>::value;
}

template<typename T, typename A, typename B>
using enable_if_either = typename std::enable_if<is_either<T, A, B>(), T>::type;

template<typename T>
using enable_if_int = enable_if_either<T, unsigned, int>;

template<typename T>
using enable_if_long = enable_if_either<T, unsigned long, long>;

template<typename T>
using enable_if_long_long = enable_if_either<T, unsigned long long, long long>;

template<typename T>
inline unsigned findFirstBit(enable_if_int<T> Value) {
  return ffs(Value);
}

template<typename T>
inline unsigned findFirstBit(enable_if_long<T> Value) {
  return ffsl(Value);
}

template<typename T>
inline unsigned findFirstBit(enable_if_long_long<T> Value) {
  return ffsll(Value);
}

template<typename T>
inline unsigned findFirstBit(T Value) {
  return findFirstBit<T>(Value);
}

template<typename T>
inline T excessDivide(T A, unsigned B) {
  return (A + (B - 1)) / B;
}

class LazySmallBitVector;

template<typename LSBV>
class LazySmallBitVectorIterator
  : public boost::iterator_facade<LazySmallBitVectorIterator<LSBV>,
                                  unsigned,
                                  boost::forward_traversal_tag,
                                  unsigned> {
public:
  LazySmallBitVectorIterator() : BitVector(nullptr), NextBitIndex(0) {}
  LazySmallBitVectorIterator(LSBV *BitVector);
  LazySmallBitVectorIterator(LSBV *BitVector, unsigned Index);

private:
  void increment();

  bool equal(LazySmallBitVectorIterator const &Other) const {
    return BitVector == Other.BitVector && NextBitIndex == Other.NextBitIndex;
  }

  unsigned dereference() const {
    revng_assert(BitVector != nullptr && NextBitIndex != 0);
    return NextBitIndex - 1;
  }

private:
  friend class boost::iterator_core_access;

  LSBV *BitVector;
  unsigned NextBitIndex;
};

/// \brief Infinite zero-initialized BitVector, automatically enlarging and
///        in-place up to sizeof(uintptr_t) * CHAR_BIT - 1 bits
class LazySmallBitVector {
public:
  using const_iterator = LazySmallBitVectorIterator<const LazySmallBitVector>;
  using iterator = LazySmallBitVectorIterator<LazySmallBitVector>;
  typedef bool value_type;

private:
  static const unsigned BitsPerPointer = sizeof(uintptr_t) * CHAR_BIT;
  static const unsigned MaxSmallSize = BitsPerPointer - 1;
  static const uintptr_t One = 1;
  static const unsigned IntMax = std::numeric_limits<int32_t>::max();

  struct LargeStorage {
    unsigned wordCount() const { return Capacity / BitsPerPointer; }
    unsigned capacity() const { return Capacity; }

    unsigned requiredBits() const {
      for (signed I = wordCount() - 1; I >= 0; I--)
        if (at(I) != 0)
          return BitsPerPointer * I + ::requiredBits(at(I));

      return 0;
    }

    uintptr_t &at(size_t Index) {
      revng_assert(Index < wordCount());
      return Storage[Index];
    }

    const uintptr_t &at(size_t Index) const {
      revng_assert(Index < wordCount());
      return Storage[Index];
    }

    void zero(size_t From, size_t Count) {
      revng_assert(From + Count <= wordCount());
      memset(&at(From), 0, Count * sizeof(uintptr_t));
    }

    void zero(size_t From) { zero(From, wordCount()); }

    void zero() { zero(0); }

    void setCapacity(size_t Count) { Capacity = Count; }

    LargeStorage &operator=(const LargeStorage &Other) {
      revng_assert(Capacity >= Other.Capacity);
      memcpy(&at(0), &Other.at(0), Other.Capacity / sizeof(uintptr_t));
      return *this;
    }

  private:
    size_t Capacity;
    uintptr_t Storage[1];
  };

public:
  LazySmallBitVector() : Storage(1) {}

  LazySmallBitVector(const LazySmallBitVector &Other) : Storage(1) {
    *this = Other;
  }

  LazySmallBitVector(LazySmallBitVector &&Other) : Storage(Other.Storage) {
    Other.Storage = 1;
  }

  bool isSmall() const { return Storage & 1; }

  void set(unsigned Index) {
    revng_assert(Index < IntMax);

    if (Index >= capacity())
      alloc(Index + 1);

    if (isSmall()) {
      revng_assert(Index < MaxSmallSize);
      Storage = Storage | (One << (Index + 1));
    } else {
      uintptr_t &Target = getLarge().at(Index / BitsPerPointer);
      Target = Target | (One << (Index % BitsPerPointer));
    }
  }

  void unset(unsigned Index) {
    revng_assert(Index < IntMax);

    if (Index >= capacity())
      return;

    if (isSmall()) {
      revng_assert(Index < MaxSmallSize);
      Storage = Storage & ~(One << (Index + 1));
    } else {
      uintptr_t &Target = getLarge().at(Index / BitsPerPointer);
      Target = Target & ~(One << (Index % BitsPerPointer));
    }
  }

  void zero(size_t From, size_t Count) {
    if (From >= capacity() || From + Count > capacity())
      return;

    if (isSmall()) {
      setSmall(getSmall() & ~(((One << Count) - 1) << From));
    } else {
      LargeStorage &Large = getLarge();

      // Blank leading bits
      unsigned Head = BitsPerPointer - (From % BitsPerPointer);
      unsigned HeadComplement = From % BitsPerPointer;
      if (HeadComplement != 0) {
        uintptr_t Mask = ~(((One << Head) - 1) << HeadComplement);
        Large.at(From / BitsPerPointer) &= Mask;
        From += Head;
        Count -= Head;
        revng_assert(From % BitsPerPointer == 0);
      }

      // Blank trailing bits
      unsigned Tail = Count % BitsPerPointer;
      if (Tail != 0) {
        Large.at((From + Count) / BitsPerPointer) &= ~((One << Tail) - 1);
        Count -= Tail;
        revng_assert(Count % BitsPerPointer == 0);
      }

      if (Count != 0)
        Large.zero(From / BitsPerPointer, Count / BitsPerPointer);
    }
  }

  void zero(size_t From) { zero(From, capacity() - From); }

  void zero() { zero(0); }

  bool operator[](unsigned Index) const {
    if (Index >= capacity())
      return false;

    if (isSmall()) {
      return (Storage >> (Index + 1)) & 1;
    } else {
      uintptr_t Value = getLarge().at(Index / BitsPerPointer);
      return (Value >> (Index % BitsPerPointer)) & 1;
    }
  }

  /// \brief The bits required to represent this bit vector
  ///
  /// This is the index of the most significant set bit, plus 1. 0 means the bit
  /// vector is composed exclusively by zeros.
  unsigned requiredBits() const {
    if (isSmall()) {
      uintptr_t Value = getSmall();
      return ::requiredBits(Value);
    } else {
      return getLarge().requiredBits();
    }
  }

  bool isZero() const { return requiredBits() == 0; }

  LazySmallBitVector &operator=(const LazySmallBitVector &Other) {
    if (!(Other.isSmall() || Other.capacity() > 63))
      revng_abort();

    if (Other.isSmall()) {
      Storage = Other.Storage;
    } else {
      if (Other.capacity() > capacity())
        alloc(Other.capacity());

      getLarge() = Other.getLarge();
    }

    return *this;
  }

  LazySmallBitVector &operator=(LazySmallBitVector &&Other) {
    if (!isSmall())
      free(&getLarge());

    Storage = Other.Storage;

    Other.Storage = 1;

    return *this;
  }

  bool operator==(const LazySmallBitVector &Other) const {
    if (Storage == Other.Storage)
      return true;

    if (isSmall() && Other.isSmall())
      return false;

    if (!isSmall() && !Other.isSmall()) {
      const LargeStorage &OtherLarge = Other.getLarge();
      const LargeStorage &ThisLarge = getLarge();

      unsigned Max = std::min(ThisLarge.capacity(), OtherLarge.capacity());
      Max /= BitsPerPointer;

      for (unsigned I = 0; I < Max; I++)
        if (ThisLarge.at(I) != OtherLarge.at(I))
          return false;

      if (ThisLarge.capacity() > OtherLarge.capacity()) {
        for (unsigned I = Max; I < ThisLarge.wordCount(); I++)
          if (ThisLarge.at(I) != 0)
            return false;
      } else {
        for (unsigned I = Max; I < OtherLarge.wordCount(); I++)
          if (OtherLarge.at(I) != 0)
            return false;
      }

    } else if (!isSmall() && Other.isSmall()) {
      const LargeStorage &ThisLarge = getLarge();
      if (ThisLarge.at(0) != Other.getSmall())
        return false;

      for (unsigned I = 1; I < ThisLarge.wordCount(); I++)
        if (ThisLarge.at(I) != 0)
          return false;
    } else if (isSmall() && !Other.isSmall()) {
      const LargeStorage &OtherLarge = Other.getLarge();
      if (OtherLarge.at(0) != getSmall())
        return false;

      for (unsigned I = 1; I < OtherLarge.wordCount(); I++)
        if (OtherLarge.at(I) != 0)
          return false;
    }

    return true;
  }

  bool operator!=(const LazySmallBitVector &Other) const {
    return !(*this == Other);
  }

  LazySmallBitVector &operator^=(const LazySmallBitVector &Other) {
    // Ensure we have at least the same capacity as Other
    if (Other.capacity() > this->capacity())
      alloc(Other.capacity());

    // This situation should never happen, since we just ensured we have at
    // least the same capacity
    revng_assert(!(isSmall() && !Other.isSmall()));

    if (isSmall() && Other.isSmall()) {
      Storage = (Storage ^ Other.Storage) | 1;
    } else if (!isSmall() && !Other.isSmall()) {
      const LargeStorage &OtherLarge = Other.getLarge();
      LargeStorage &ThisLarge = getLarge();

      unsigned Max = std::min(ThisLarge.capacity(), OtherLarge.capacity());
      Max /= BitsPerPointer;
      for (unsigned I = 0; I < Max; I++)
        ThisLarge.at(I) = ThisLarge.at(I) ^ OtherLarge.at(I);

    } else if (!isSmall() && Other.isSmall()) {
      LargeStorage &ThisLarge = getLarge();
      ThisLarge.at(0) = ThisLarge.at(0) ^ Other.getSmall();
    }

    return *this;
  }

  LazySmallBitVector &operator|=(const LazySmallBitVector &Other) {
    // Ensure we have at least the same capacity as Other
    if (Other.capacity() > this->capacity())
      alloc(Other.capacity());

    // This situation should never happen, since we just ensured we have at
    // least the same capacity
    revng_assert(!(isSmall() && !Other.isSmall()));

    if (isSmall() && Other.isSmall()) {
      Storage = Storage | Other.Storage;
    } else if (!isSmall() && !Other.isSmall()) {
      const LargeStorage &OtherLarge = Other.getLarge();
      LargeStorage &ThisLarge = getLarge();

      unsigned Max = std::min(ThisLarge.capacity(), OtherLarge.capacity());
      Max /= BitsPerPointer;

      for (unsigned I = 0; I < Max; I++)
        ThisLarge.at(I) = ThisLarge.at(I) | OtherLarge.at(I);

    } else if (!isSmall() && Other.isSmall()) {
      LargeStorage &ThisLarge = getLarge();
      ThisLarge.at(0) = ThisLarge.at(0) | Other.getSmall();
    }

    return *this;
  }

  LazySmallBitVector &operator&=(const LazySmallBitVector &Other) {
    if (isSmall()) {
      uintptr_t OtherValue;
      if (Other.isSmall())
        OtherValue = Other.getSmall();
      else
        OtherValue = Other.getLarge().at(0);

      setSmall(getSmall() & OtherValue);
    } else {
      LargeStorage &Large = getLarge();
      size_t ThisPointersCount = Large.wordCount();

      if (Other.isSmall()) {
        // We have to discard everything except the first uintptr_t
        if (ThisPointersCount > 1) {
          // Zero out all the uintptr_t after the first one
          Large.zero(1, ThisPointersCount - 1);
        }

        Large.at(0) = Large.at(0) & Other.getSmall();
      } else {
        const LargeStorage &OtherLarge = Other.getLarge();
        size_t OtherPointersCount = OtherLarge.wordCount();

        if (ThisPointersCount > OtherPointersCount) {
          // Zero out all the uintptr_t after OtherPointersCount
          Large.zero(OtherPointersCount,
                     ThisPointersCount - OtherPointersCount);
        }

        unsigned Max = std::min(OtherPointersCount, ThisPointersCount);
        for (unsigned I = 0; I < Max; I++)
          Large.at(I) = Large.at(I) & OtherLarge.at(I);
      }
    }

    return *this;
  }

  LazySmallBitVector &operator>>=(unsigned Amount) {
    revng_assert(Amount <= capacity());

    if (isSmall()) {
      Storage >>= Amount;
      Storage |= 1;
      return *this;
    }

    LargeStorage &Large = getLarge();

    if (Amount == Large.capacity()) {
      Large.zero();
      return *this;
    }

    unsigned SourceIndex = Amount / BitsPerPointer;
    unsigned Count = Large.wordCount() - SourceIndex;

    auto Destination = [&Large](unsigned I) -> uintptr_t & {
      return Large.at(I);
    };

    auto Source = [SourceIndex, &Large](unsigned I) -> uintptr_t & {
      return Large.at(SourceIndex + I);
    };

    unsigned Bits = Amount % BitsPerPointer;
    unsigned OtherBits = BitsPerPointer - Bits;

    // Shift all the word except the last one
    unsigned I = 0;
    for (; I < Count - 1; I++)
      Destination(I) = Source(I) >> Bits | (Source(I + 1) << OtherBits);

    Destination(I) = Source(I) >> Bits;

    if (SourceIndex > 0)
      Large.zero(I + 1);

    return *this;
  }

  LazySmallBitVector &operator<<=(unsigned Amount) {
    // Get the current amount of bits
    unsigned RequiredBits = requiredBits();

    if (isSmall()) {
      unsigned NewSize = RequiredBits + Amount;

      if (NewSize <= BitsPerPointer) {
        // We fit where we are
        Storage = Storage & ~1;
        Storage <<= Amount;
        Storage |= 1;
        return *this;
      } else {
        // We have to enlarge
        alloc(NewSize);
      }
    }

    LargeStorage &Large = getLarge();

    // Is it all zeros? Do nothing.
    if (RequiredBits == 0)
      return *this;

    // Enlarge, if required
    RequiredBits += Amount;
    if (RequiredBits > Large.capacity())
      alloc(RequiredBits + 1);

    LargeStorage &NewLarge = getLarge();

    unsigned ToSkip = Amount / BitsPerPointer;
    auto Destination = [ToSkip, &NewLarge](unsigned I) -> uintptr_t & {
      return NewLarge.at(ToSkip + I);
    };

    auto Source = [&NewLarge](unsigned I) -> uintptr_t & {
      return NewLarge.at(I);
    };

    unsigned Bits = Amount % BitsPerPointer;
    unsigned OtherBits = BitsPerPointer - Bits;
    for (signed I = NewLarge.wordCount() - 1 - ToSkip; I >= 0 + 1; I--)
      Destination(I) = (Source(I) << Bits) | (Source(I - 1) >> OtherBits);

    Destination(0) = Source(0) << Bits;

    if (ToSkip > 0)
      NewLarge.zero(0, ToSkip);

    return *this;
  }

  /// \brief Returns the 1-based index of the next set bit after \p StartIndex
  ///
  /// \return 0 if no bits are set after \p StartIndex, the 1-based index of the
  ///         next bit set otherwise
  unsigned findNext(unsigned StartIndex) const {
    if (StartIndex >= requiredBits())
      return 0;

    if (isSmall()) {
      return StartIndex + findFirstBit(getSmall() >> StartIndex);
    } else {
      const LargeStorage &Large = getLarge();
      unsigned Index = StartIndex / BitsPerPointer;
      unsigned ShiftAmount = StartIndex % BitsPerPointer;

      uintptr_t FirstValue = Large.at(Index) >> ShiftAmount;
      if (FirstValue != 0)
        return StartIndex + findFirstBit(FirstValue);

      do {
        Index++;
        if (Index * BitsPerPointer >= capacity())
          return 0;
      } while (Large.at(Index) == 0);

      return Index * BitsPerPointer + findFirstBit(Large.at(Index));
    }
  }

  const_iterator begin() const { return const_iterator(this); }
  const_iterator end() const { return const_iterator(this, 0); }

  iterator begin() { return iterator(this); }
  iterator end() { return iterator(this, 0); }

  ~LazySmallBitVector() {
    if (!isSmall())
      free(&getLarge());
    Storage = 0;
  }

private:
  friend iterator;
  friend const_iterator;

  uintptr_t getSmall() const {
    revng_assert(isSmall());
    return Storage >> 1;
  }

  void setSmall(uintptr_t Value) {
    revng_assert(isSmall());
    Storage = (Value << 1) | 1;
    revng_assert(isSmall());
  }

  size_t capacity() const {
    if (isSmall())
      return BitsPerPointer - 1;
    else
      return getLarge().capacity();
  }

  void alloc(size_t NewSize) {
    revng_assert(NewSize > capacity());

    // Allocate the maximum between the requested index and twice the current
    // capacity (rounding to the size of a uintptr_t)
    size_t RequestedBits = std::max(NewSize, 2 * capacity());
    size_t PointersCount = excessDivide(RequestedBits, BitsPerPointer);

    // The `- 1` is due to the fact that LargeStorage already includes an
    // uintptr_t entry
    size_t ExtraSize = (PointersCount - 1) * sizeof(uintptr_t);
    void *Ptr = malloc(sizeof(LargeStorage) + ExtraSize);
    revng_assert(Ptr != nullptr);
    LargeStorage &Result = *reinterpret_cast<LargeStorage *>(Ptr);

    // Initialize the Capacity field
    Result.setCapacity(PointersCount * BitsPerPointer);

    // Zero out the storage
    Result.zero();

    // Copy the old values into the new storage
    if (isSmall()) {
      Result.at(0) = getSmall();
    } else {
      LargeStorage &Old = getLarge();
      Result = Old;

      // Also, deallocate the old storage
      free(&Old);
    }

    Storage = reinterpret_cast<uintptr_t>(&Result);
    revng_assert(!isSmall());
  }

  LargeStorage &getLarge() {
    revng_assert(!isSmall());
    return *reinterpret_cast<LargeStorage *>(Storage);
  }

  const LargeStorage &getLarge() const {
    revng_assert(!isSmall());
    return *reinterpret_cast<const LargeStorage *>(Storage);
  }

private:
  uintptr_t Storage;
};

template<typename LSBV>
inline void LazySmallBitVectorIterator<LSBV>::increment() {
  revng_assert(BitVector != nullptr);
  revng_assert(NextBitIndex == 0 || (*BitVector)[NextBitIndex - 1] == true);

  NextBitIndex = BitVector->findNext(NextBitIndex);
}

#define LSBVI LazySmallBitVectorIterator

template<typename LSBV>
inline LSBVI<LSBV>::LSBVI(LSBV *BitVector) :
  BitVector(BitVector),
  NextBitIndex(0) {

  revng_assert(BitVector != nullptr);
  if (!BitVector->isZero())
    increment();
}

template<typename LSBV>
inline LSBVI<LSBV>::LSBVI(LSBV *BitVector, unsigned Index) :
  BitVector(BitVector),
  NextBitIndex(Index) {
  revng_assert(BitVector != nullptr);
}

#undef LSBVI

#endif // LAZYSMALLBITVECTOR_H
