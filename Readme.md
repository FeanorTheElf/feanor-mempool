# feanor-mempool

A simple memory manager interface that I designed for use in my math library [feanor-math](https://github.com/feanortheelf/feanor-math).
Mainly, it defines the trait `MemoryProvider<T>` for objects that can provide homogeneous arrays containing elements of size `T`.
Standard implementations could use basic heap allocation, or for example caching/"recycling" of used memory. 
This allows making algorithms or structs that require frequent allocations configurable, by giving them a suitable memory provider.

## Design rationale

I wanted a simple memory provider library with the following features:
 - Interchangeability of standard allocations and pooled/cached allocations
 - Type erasure of the type of the memory pool. 
   My earlier implementation required every user of a memory provider to be generic with respect to its type, which added (unnecessary ?) complexity to the code.
 - Memory returned by a memory pool should have a dynamically managed reference to the memory pool (i.e. no lifetime parameter).
   This requires the use of shared pointers, but leads to much simpler type management.

To summarize, I accept slightly less performant options to achieve points 2 and 3.
In feanor-math however, the use of large arrays usually goes hand in hand with significant computational cost, so the memory management cost should be negligible.

## Example

```rust
use feanor_mempool::*;

fn do_something(len: usize, mem_provider: MemProviderRc<i32>) {
    for _ in 0..10 {
        // the memory is possible reused from the previous loop (when mem_provider is a `SizedMemoryPool`)
        let data = mem_provider.new_init(len, |i| i as i32);
        for i in 0..len {
            assert_eq!(i as i32, data[i]);
        }
    }
}

do_something(1000, cache::SizedMemoryPool::new_dynamic_size(1));
do_something(1000, default_memory_provider!());
```