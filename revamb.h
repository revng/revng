/* TODO: improve error handling */

enum class DebugInfoType {
  None,
  OriginalAssembly,
  PTC
};

class Architecture {
public:
  Architecture() : PCReg("pc"), DefaultAlignment(1), Endianess(LittleEndian) {}

  enum EndianessType {
    LittleEndian,
    BigEndian
  };

  unsigned defaultAlignment() { return DefaultAlignment; }
  EndianessType endianess() { return Endianess; }
  const std::string PCReg;

private:
  unsigned DefaultAlignment;
  EndianessType Endianess;
};

// TODO: this requires C++14
template<typename FnTy, FnTy Ptr>
struct GenericFunctor {
  template<typename... Args>
  auto operator()(Args... args) {
    return Ptr(args...);
  }
};
