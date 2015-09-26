/* TODO: improve error handling */

class Architecture {
public:
  Architecture() : default_alignment(1), endianess(LittleEndian) {}

  enum EndianessType {
    LittleEndian,
    BigEndian
  };

  unsigned DefaultAlignment() { return default_alignment; }
  EndianessType Endianess() { return endianess; }

private:
  unsigned default_alignment;
  EndianessType endianess;
};

// TODO: this requires C++14
template<typename FnTy, FnTy Ptr>
struct GenericFunctor {
  template<typename... Args>
  auto operator()(Args... args) {
    return Ptr(args...);
  }
};
