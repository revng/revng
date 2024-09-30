#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"

// Souped-up llvm::ArrayRef which overloads for both uint8_t and char "array"
// datastructures
class DataBuffer {
private:
  const uint8_t *Ptr = nullptr;
  size_t Size = 0;

public:
  template<typename T>
    requires(std::is_same_v<T, char> or std::is_same_v<T, uint8_t>)
  DataBuffer(const T *Ptr, size_t Size) :
    Ptr(reinterpret_cast<const uint8_t *>(Ptr)), Size(Size) {}

  DataBuffer(const llvm::MemoryBuffer &Buf) :
    Ptr(reinterpret_cast<const uint8_t *>(Buf.getBufferStart())),
    Size(Buf.getBufferSize()) {}

  DataBuffer(const llvm::StringRef &Ref) :
    Ptr(reinterpret_cast<const uint8_t *>(Ref.data())), Size(Ref.size()) {}

  DataBuffer(llvm::ArrayRef<char> &Ref) :
    Ptr(reinterpret_cast<const uint8_t *>(Ref.data())), Size(Ref.size()) {}

  DataBuffer(llvm::ArrayRef<uint8_t> &Ref) :
    Ptr(Ref.data()), Size(Ref.size()) {}

  template<typename T, typename U>
    requires(std::is_same_v<T, char> or std::is_same_v<T, uint8_t>)
  DataBuffer(const llvm::SmallVectorTemplateCommon<T, U> &Vec) :
    Ptr(reinterpret_cast<const uint8_t *>(Vec.data())), Size(Vec.size()) {}

public:
  const uint8_t *data() { return Ptr; }
  size_t size() { return Size; }
  llvm::ArrayRef<uint8_t> arrayRef() { return { Ptr, Size }; }
};
