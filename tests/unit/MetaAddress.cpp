/// \file MetaAddress.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#define BOOST_TEST_MODULE MetaAddress
bool init_unit_test();
#include "boost/test/execution_monitor.hpp"
#include "boost/test/unit_test.hpp"

#include "revng/Support/MetaAddress.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

BOOST_TEST_DONT_PRINT_LOG_VALUE(MetaAddress)
BOOST_TEST_DONT_PRINT_LOG_VALUE(MetaAddressType::Values)

using namespace llvm;
using namespace model::Architecture;

static MetaAddress generic32(uint64_t Address) {
  return MetaAddress::fromGeneric(x86, Address);
}

static MetaAddress generic64(uint64_t Address) {
  return MetaAddress::fromGeneric(x86_64, Address);
}

static MetaAddress pc(uint64_t Address) {
  return MetaAddress::fromPC(x86, Address);
}

BOOST_AUTO_TEST_CASE(Constructor) {
  using namespace MetaAddressType;

  BOOST_TEST(MetaAddress().isInvalid());
  BOOST_TEST(MetaAddress(0x1000, Generic32).isValid());
  BOOST_TEST(MetaAddress(0x1000, Generic32).type() == Generic32);
  BOOST_TEST(MetaAddress(0x1000, Generic32).address() == uint64_t(0x1000));

  BOOST_TEST(MetaAddress(0x1000, Code_arm).isValid());
  BOOST_TEST(MetaAddress(0x1001, Code_arm).isInvalid());
  BOOST_TEST(MetaAddress(0x1000, Code_arm_thumb).isValid());
  BOOST_TEST(MetaAddress(0x1001, Code_arm_thumb).isInvalid());
}

BOOST_AUTO_TEST_CASE(Factory) {
  // Invalid
  BOOST_TEST(MetaAddress::invalid().isInvalid());

  // Code
  BOOST_TEST(pc(0x1000).isCode());
  BOOST_TEST(pc(0x1000).isCode(x86));

  // Regular ARM and Thumb are both ARM
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1000).isCode(arm));
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1001).isCode(arm));

  // Generic
  BOOST_TEST(generic64(0x1000).isGeneric());

  // Convert to generic
  BOOST_TEST(pc(0x1000).toGeneric().isGeneric());

  // bitSize
  BOOST_TEST(MetaAddress::fromPC(arm, 0).bitSize() == uint64_t(32));
  BOOST_TEST(MetaAddress::fromPC(aarch64, 0).bitSize() == uint64_t(64));

  // Epoch
  BOOST_TEST(generic64(0).epoch() == uint64_t(0));
  BOOST_TEST(generic64(0).isDefaultEpoch());

  // Address space
  BOOST_TEST(generic64(0).addressSpace() == uint64_t(0));
  BOOST_TEST(generic64(0).isDefaultAddressSpace());
}

BOOST_AUTO_TEST_CASE(Accessors) {
  BOOST_TEST(MetaAddress::invalid().asPCOrZero() == uint64_t(0));
  BOOST_TEST(pc(0x1000).asPC() == uint64_t(0x1000));
  using MA = MetaAddress;
  BOOST_TEST(MA::fromPC(arm, 0x1001).asPC() == uint64_t(0x1001));
  BOOST_TEST(generic64(0x1000).address() == uint64_t(0x1000));
}

BOOST_AUTO_TEST_CASE(Arithmetic) {
  BOOST_TEST(generic64(0x1000) + 1 == generic64(0x1001));
  BOOST_TEST(generic64(0x1001) - 1 == generic64(0x1000));
  BOOST_TEST(generic64(0x1000) + 0x1000 == generic64(0x2000));

  auto Distance = generic64(0x1010) - generic64(0x1000);
  BOOST_TEST(Distance.has_value());
  BOOST_TEST((*Distance) == uint64_t(0x10));

  Distance = generic64(0x1000) - generic64(0x1010);
  BOOST_TEST(not Distance.has_value());
}

BOOST_AUTO_TEST_CASE(Overflow) {
  BOOST_TEST(generic32(0xFFFFFFFF) + 1 == MetaAddress::invalid());
  BOOST_TEST(generic64(0xFFFFFFFF) + 1 == generic64(0x100000000));
  BOOST_TEST(generic64(0xFFFFFFFFFFFFFFFF) + 1 == MetaAddress::invalid());
  BOOST_TEST(generic64(0) - 1 == MetaAddress::invalid());
  BOOST_TEST(generic32(0) - 1 == MetaAddress::invalid());
}

BOOST_AUTO_TEST_CASE(Thumb) {
  auto NonThumb = MetaAddress::fromPC(arm, 0x1000);
  BOOST_TEST(NonThumb.type() == MetaAddressType::Code_arm);
  BOOST_TEST(NonThumb.address() == uint64_t(0x1000));
  BOOST_TEST(NonThumb.asPC() == uint64_t(0x1000));

  auto Thumb = MetaAddress::fromPC(arm, 0x1001);
  BOOST_TEST(Thumb.type() == MetaAddressType::Code_arm_thumb);
  BOOST_TEST(Thumb.address() == uint64_t(0x1000));
  BOOST_TEST(Thumb.asPC() == uint64_t(0x1001));
}

BOOST_AUTO_TEST_CASE(Alignment) {
  // Regular ARM
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1000).isValid());

  // Thumb aligned at 4-bytes
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1001).isValid());

  // Thumb aligned at 2-bytes
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1003).isValid());

  // Misaligned regular ARM
  BOOST_TEST(MetaAddress::fromPC(arm, 0x1002).isInvalid());

  // MIPS
  BOOST_TEST(MetaAddress::fromPC(mips, 0x1000).isValid());
  BOOST_TEST(MetaAddress::fromPC(mips, 0x1001).isInvalid());
  BOOST_TEST(MetaAddress::fromPC(mips, 0x1002).isInvalid());
  BOOST_TEST(MetaAddress::fromPC(mips, 0x1003).isInvalid());

  // x86
  BOOST_TEST(MetaAddress::fromPC(x86, 0x1000).isValid());
  BOOST_TEST(MetaAddress::fromPC(x86, 0x1001).isValid());
  BOOST_TEST(MetaAddress::fromPC(x86, 0x1002).isValid());
  BOOST_TEST(MetaAddress::fromPC(x86, 0x1003).isValid());

  // SystemZ
  BOOST_TEST(MetaAddress::fromPC(systemz, 0x1000).isValid());
  BOOST_TEST(MetaAddress::fromPC(systemz, 0x1001).isInvalid());
  BOOST_TEST(MetaAddress::fromPC(systemz, 0x1002).isValid());
}

BOOST_AUTO_TEST_CASE(Comparison) {
  auto A = MetaAddress::fromGeneric(x86, 0x1000);
  auto B = MetaAddress::fromGeneric(x86, 0x1001);

  BOOST_TEST(A.addressLowerThan(B));
  BOOST_TEST(A != B);
  BOOST_TEST(!(A.addressGreaterThan(B)));

  A += 1;
  BOOST_TEST(!(A.addressLowerThan(B)));
  BOOST_TEST(A == B);
  BOOST_TEST(!(A.addressGreaterThan(B)));

  A += 1;
  BOOST_TEST(!(A.addressLowerThan(B)));
  BOOST_TEST(A != B);
  BOOST_TEST(A.addressGreaterThan(B));
}

BOOST_AUTO_TEST_CASE(Page) {
  BOOST_TEST(generic64(0x1234).pageStart() == generic64(0x1000));
  BOOST_TEST(generic64(0x1234).nextPageStart() == generic64(0x2000));
}

BOOST_AUTO_TEST_CASE(Map) {
  std::map<MetaAddress, int> Map;

  Map[generic64(0)] = 1;
  Map[MetaAddress::invalid()] = 1;
  Map[pc(0)] = 1;
  Map[MetaAddress::fromPC(arm, 0)] = 1;
  Map[MetaAddress::fromPC(arm, 1)] = 1;

  BOOST_TEST(Map.size() == size_t(5));
}
