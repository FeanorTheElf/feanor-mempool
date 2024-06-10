# feanor-mempool

Previously, this was a simple memory manger interface that I designed for use in my math library [feanor-math](https://github.com/feanortheelf/feanor-math).
However, now I decided to instead use the allocator API that is available in Rust nightly.
Hence, this library offers `std::alloc::Allocator`s designed for use with feanor-math.
