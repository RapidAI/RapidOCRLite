#include "onnx_min.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ppocr::indra {
namespace {

struct Reader {
  const uint8_t* p;
  const uint8_t* e;
  bool done() const { return p >= e; }
  uint64_t varint() {
    uint64_t v = 0;
    int s = 0;
    while (p < e) {
      uint8_t b = *p++;
      v |= uint64_t(b & 0x7f) << s;
      if (!(b & 0x80)) return v;
      s += 7;
    }
    throw std::runtime_error("indra onnx: truncated varint");
  }
  Reader sub() {
    uint64_t n = varint();
    if (uint64_t(e - p) < n) throw std::runtime_error("indra onnx: truncated field");
    Reader r{p, p + n};
    p += n;
    return r;
  }
  void skip(int wt) {
    if (wt == 0) varint();
    else if (wt == 1) p += 8;
    else if (wt == 2) sub();
    else if (wt == 5) p += 4;
    else throw std::runtime_error("indra onnx: bad wire type");
  }
  std::string str() {
    Reader r = sub();
    return std::string(reinterpret_cast<const char*>(r.p), r.e - r.p);
  }
};

OTensor ParseTensor(Reader r, std::string* name) {
  OTensor t;
  std::string raw;
  while (!r.done()) {
    uint64_t key = r.varint();
    int f = int(key >> 3), wt = int(key & 7);
    if (f == 1) {
      if (wt == 2) { Reader s = r.sub(); while (!s.done()) t.dims.push_back(int64_t(s.varint())); }
      else t.dims.push_back(int64_t(r.varint()));
    } else if (f == 2) t.dtype = int(r.varint());
    else if (f == 4 && wt == 2) {
      Reader s = r.sub();
      while (s.p + 4 <= s.e) { float v; std::memcpy(&v, s.p, 4); s.p += 4; t.f.push_back(v); }
    } else if (f == 7 && wt == 2) { Reader s = r.sub(); while (!s.done()) t.i.push_back(int64_t(s.varint())); }
    else if (f == 8) { std::string n = r.str(); if (name) *name = n; }
    else if (f == 9) raw = r.str();
    else r.skip(wt);
  }
  if (!raw.empty()) {
    if (t.dtype == 1) { t.f.resize(raw.size() / 4); std::memcpy(t.f.data(), raw.data(), t.f.size() * 4); }
    else if (t.dtype == 7) { t.i.resize(raw.size() / 8); std::memcpy(t.i.data(), raw.data(), t.i.size() * 8); }
    else if (t.dtype == 9) { for (char c : raw) t.i.push_back(c ? 1 : 0); }
  }
  return t;
}

void ParseAttr(Reader r, ONode& n) {
  std::string name;
  OAttr a;
  while (!r.done()) {
    uint64_t key = r.varint();
    int f = int(key >> 3), wt = int(key & 7);
    if (f == 1) name = r.str();
    else if (f == 2 && wt == 5) { std::memcpy(&a.f, r.p, 4); r.p += 4; }
    else if (f == 3) a.i = int64_t(r.varint());
    else if (f == 5) { a.t = ParseTensor(r.sub(), nullptr); a.has_t = true; }
    else if (f == 8) {
      if (wt == 2) { Reader s = r.sub(); while (!s.done()) a.ints.push_back(int64_t(s.varint())); }
      else a.ints.push_back(int64_t(r.varint()));
    } else r.skip(wt);
  }
  n.attr[name] = std::move(a);
}

}  // namespace

OModel LoadOnnx(const std::string& path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("indra onnx: cannot open " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(fs)), {});
  OModel m;
  Reader top{buf.data(), buf.data() + buf.size()};
  while (!top.done()) {
    uint64_t key = top.varint();
    int f = int(key >> 3), wt = int(key & 7);
    if (f != 7) { top.skip(wt); continue; }
    Reader g = top.sub();
    while (!g.done()) {
      uint64_t k2 = g.varint();
      int f2 = int(k2 >> 3), w2 = int(k2 & 7);
      if (f2 == 1) {
        Reader nr = g.sub();
        ONode n;
        while (!nr.done()) {
          uint64_t k3 = nr.varint();
          int f3 = int(k3 >> 3), w3 = int(k3 & 7);
          if (f3 == 1) n.in.push_back(nr.str());
          else if (f3 == 2) n.out.push_back(nr.str());
          else if (f3 == 4) n.op = nr.str();
          else if (f3 == 5) ParseAttr(nr.sub(), n);
          else nr.skip(w3);
        }
        m.nodes.push_back(std::move(n));
      } else if (f2 == 5) {
        std::string name;
        OTensor t = ParseTensor(g.sub(), &name);
        m.init[name] = std::move(t);
      } else g.skip(w2);
    }
  }
  return m;
}

}  // namespace ppocr::indra
