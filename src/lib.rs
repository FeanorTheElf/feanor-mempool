#![feature(never_type)]
#![feature(new_uninit)]
#![feature(maybe_uninit_slice)]

#![doc = include_str!("../Readme.md")]

pub mod cache;

use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::rc::*;

///
/// Trait for memory regions managed by a memory provider.
/// 
/// Note that the memory content of such an object does not have to be
/// initialized. Thus, when implementing it, make sure sure the memory
/// itself is declared as `[MaybeUninit<T>]` or similar, instead of `[T]`.
/// 
/// When any of the functions [`MemoryProviderObject::drop_or_recycle()`],
/// [`MemoryProviderObject::deref_initialized()`] or [`MemoryProviderObject::deref_mut_initialized()`]
/// are called, you can assume that the contained data has been initialized.
/// 
pub trait MemoryProviderObject<T> {

    ///
    /// Returns the data as [`MaybeUninit`]-slice.
    /// 
    fn deref_uninit(&mut self) -> &mut [MaybeUninit<T>];

    ///
    /// Drops the contained data and performs any cleanup that is
    /// necessary when the memory is no longer required. This may
    /// mean deallocating it, or returning it to some memory pool.
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the contained data is initialized
    /// and valid to be dropped.
    /// 
    unsafe fn drop_or_recycle(self: Box<Self>);

    ///
    /// Returns the data as constant `T`-slice.
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the contained data is initialized.
    /// 
    unsafe fn deref_initialized(&self) -> &[T];

    ///
    /// Returns the data as mutable `T`-slice.
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the contained data is initialized.
    /// 
    unsafe fn deref_mut_initialized(&mut self) -> &mut [T];
}

///
/// A homogeneous array containing elements of type `T`, managed 
/// by a memory provider.
/// 
pub struct ManagedSlice<'a, T> {
    ///
    /// When this is non-`None`, we may assume that the memory stored by the
    /// [`MemoryProviderObject`] is initialized.
    /// 
    content: Option<Box<dyn 'a + MemoryProviderObject<T>>>
}

impl<'a, T> Deref for ManagedSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // safe, because by contract of `content`, the memory is initialized
        unsafe { self.content.as_ref().unwrap().deref_initialized() }
    }
}

impl<'a, T> DerefMut for ManagedSlice<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // safe, because by contract of `content`, the memory is initialized
        unsafe { self.content.as_mut().unwrap().deref_mut_initialized() }
    }
}

impl<'a, T> Drop for ManagedSlice<'a, T> {
    fn drop(&mut self) {
        // safe, because by contract of `content`, the memory is initialized
        unsafe { self.content.take().unwrap().drop_or_recycle() }
    }
}

///
/// Fills the given slice using elements provided by `initializer`.
/// If this function returns with `Ok(())`, then the caller may assume
/// that `mem` is correctly initialized with valid elements of `T`, i.e.
/// may call [`MaybeUninit::slice_assume_init_mut()`].
/// 
/// If this function panics or returns with `Err(E)`, this is not the case,
/// and the caller has to assume that elements of `mem` remain in a state
/// that is not a valid element of type `T`.
/// 
/// Note that if `initializer` panics, this function may leak memory, i.e.
/// will not drop elements previously returned by `initializer`. If `initialier`
/// returns an `Err(E)`, all previously yielded elements are correctly dropped.
/// 
fn try_initialize<T, E, F: FnMut(usize) -> Result<T, E>>(mem: &mut [MaybeUninit<T>], mut initializer: F) -> Result<(), E> {
    let mut i = 0;
    while i < mem.len() {
        // note that this will leak memory if initializer(i) panics
        match initializer(i) {
            Ok(val) => {
                mem[i] = MaybeUninit::new(val);
                i += 1;
            },
            Err(err) => {
                // drop the previously initialized memory
                // note that this does not prevent a memory leak in the panic case
                for j in 0..i {
                    unsafe { mem[j].assume_init_drop() };
                }
                return Err(err);
            }
        };
    }
    return Ok(());
}

///
/// Trait for objects that can provide homogeneous regions of memory (i.e. containing
/// elements of the same type `T`), often for short-term use.
/// This includes naive implementations like [`AllocatingMemoryProvider`] that
/// just allocate memory, or alternatively all kinds of memory pools and recyclers
/// (e.g. [`cache::SizedMemoryPool`]).
/// 
/// This is related to [`std::alloc::Allocator`], but less restrictive, as it may
/// return objects with certain structure. In particular, it naturally allows e.g.
/// memory pools or memory recycling.
/// 
/// It is usually used when certain objects or algorithms need frequent allocations
/// (often all of the same size), either because they need temporary, internal memory,
/// or they represent rings and have to allocate memory for elements. On the other hand,
/// if a struct just needs to store some data during its lifetime, memory pooling is
/// usually not useful, and a standard `Vec` is often used instead.
/// 
pub trait MemoryProvider<'a, T> {

    ///
    /// Provides a region of memory that is not used otherwise, of the given size.
    /// Note that the returned memory is not initialized!
    /// 
    /// # Safety
    /// 
    /// The returned memory is not initialized. In particular, this means that the implementation
    /// of the returned type of [`MemoryProviderObject`] must not store `T`s directly (e.g. in
    /// the form of [`T`]), but only `MaybeUninit<T>`. Since [`Deref`] and [`DerefMut`] change
    /// this into `&[T]` resp. `&mut [T]`, the returned value may not be dereferences until
    /// the caller has ensured proper initialization. This should be done using 
    /// [`MemoryProviderObject::deref_uninit()`].
    /// 
    /// Apart from that, it is also required that `self_rc` and `self` are references 
    /// to the same object.
    /// 
    unsafe fn new(&self, self_rc: &Rc<dyn 'a + MemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>>;
}

///
/// Shared pointer to a [`MemoryProvider`].
/// 
/// Note that it usually does not make sense to clone [`MemoryProvider`]s themselves, especially
/// when they contain a memory pool. Thus they are usually shared between many locations, which
/// can be done by [`MemProviderRc`].
/// 
pub struct MemProviderRc<'a, T> {
    pub ptr: Rc<dyn 'a + MemoryProvider<'a, T>>
}

impl<'a, T> Clone for MemProviderRc<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone()
        }
    }
}

impl<'a, T> MemProviderRc<'a, T> {
    
    pub fn new_init<F: FnMut(usize) -> T>(&self, size: usize, mut initializer: F) -> ManagedSlice<'a, T> {
        self.try_new_init::<!, _>(size, |i| Ok(initializer(i))).unwrap_or_else(|x| x)
    }

    pub fn try_new_init<E, F: FnMut(usize) -> Result<T, E>>(&self, size: usize, initializer: F) -> Result<ManagedSlice<'a, T>, E> {
        let mut data: Box<dyn 'a + MemoryProviderObject<T>> = unsafe {
            self.ptr.new(&self.ptr, size)
        };
        try_initialize(data.as_mut().deref_uninit(), initializer)?;
        return Ok(ManagedSlice { content: Some(data) });
    }
}

///
/// We can implement [`MemoryProviderObject`] for a no-op wrapper around `[T]`.
/// However, it is impossible to put this behind a trait object `dyn MemoryProviderObject<T>`,
/// probably because this would require a "super fat pointer".
/// Therefore, we have to accept the additional level of indirection.
/// 
impl<T> MemoryProviderObject<T> for Box<[MaybeUninit<T>]> {

    unsafe fn drop_or_recycle(mut self: Box<Self>) {
        for i in 0..self.len() {
            self[i].assume_init_drop();
        }
        drop(self)
    }

    unsafe fn deref_initialized(&self) -> &[T] {
        MaybeUninit::slice_assume_init_ref(self.as_ref())
    }

    unsafe fn deref_mut_initialized(&mut self) -> &mut [T] {
        MaybeUninit::slice_assume_init_mut(self.as_mut())
    }

    fn deref_uninit(&mut self) -> &mut [MaybeUninit<T>] {
        &mut *self
    }
}

///
/// Default memory provider that just allocates memory using standard allocation,
/// and drops it after it goes out of scope.
/// 
#[derive(Copy, Clone)]
pub struct AllocatingMemoryProvider;

impl<'a, T> MemoryProvider<'a, T> for AllocatingMemoryProvider
    where T: 'a
{    
    unsafe fn new(&self, _self_rc: &Rc<dyn 'a + MemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        Box::new(Box::new_uninit_slice(size))
    }
}

///
/// Memory provider that allocates memory using standard allocation, but writes
/// a log message each time.
/// 
/// The standard use case is to just use [`DefaultMemoryProvider`] everywhere, which
/// defaults to [`AllocatingMemoryProvider`]. Should you suspect repeated allocations, you can
/// toggle the feature `log_memory`, which will transparently change [`DefaultMemoryProvider`]
/// to refer to [`LoggingMemoryProvider`]. Searching the log, one can then find the locations
/// where to replace [`DefaultMemoryProvider`] with a caching variant, e.g.
/// [`cache::SizedMemoryPool`]. 
/// 
#[derive(Clone, Copy)]
pub struct LoggingMemoryProvider {
    description: &'static str
}

impl LoggingMemoryProvider {

    ///
    /// Creates a new [`LoggingMemoryProvider`] that includes the given description whenever
    /// logging an allocation.
    /// 
    pub const fn new(description: &'static str) -> Self {
        LoggingMemoryProvider { description }
    }
}

impl<'a, T> MemoryProvider<'a, T> for LoggingMemoryProvider
    where T: 'a
{    
    unsafe fn new(&self, _self_rc: &Rc<dyn 'a + MemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        println!("[{}]: Allocating {} entries of size {}", self.description, size, std::mem::size_of::<T>());
        Box::new(Box::new_uninit_slice(size))
    }
}

///
/// Default memory provider which will use standard allocation.
/// 
/// Defaults to [`AllocatingMemoryProvider`], but using the feature `log_memory`
/// it can refer to [`LoggingMemoryProvider`] instead.
/// 
#[cfg(not(feature = "log_memory"))]
pub type DefaultMemoryProvider = AllocatingMemoryProvider;

///
/// Default memory provider which will use standard allocation.
/// 
/// Defaults to [`AllocatingMemoryProvider`], but using the feature `log_memory`
/// it can refer to [`LoggingMemoryProvider`] instead.
/// 
#[cfg(feature = "log_memory")]
pub type DefaultMemoryProvider = &'static LoggingMemoryProvider;

///
/// Returns an expression that evaluates to an identifier for the current function.
/// The exact syntax of that expression is not specified, and the value should
/// only be used for debugging/tracing purposes.
/// 
#[macro_export]
macro_rules! current_function {
    () => {{
        struct LocalMemoryProvider;
        std::any::type_name::<LocalMemoryProvider>()
    }}
}

///
/// Returns an instance of the current default memory provider.
/// 
/// The used memory provider can change depending on enabled features
/// and conditional compilation.
/// 
#[macro_export]
#[cfg(not(feature = "log_memory"))]
macro_rules! default_memory_provider {
    () => {
        $crate::MemProviderRc { ptr: std::rc::Rc::new($crate::AllocatingMemoryProvider) }
    };
}

///
/// Returns an instance of the current default memory provider.
/// 
/// The used memory provider can change depending on enabled features
/// and conditional compilation.
/// 
#[macro_export]
#[cfg(feature = "log_memory")]
macro_rules! default_memory_provider {
    () => {
        $crate::MemProviderRc { ptr: std::rc::Rc::new($crate::LoggingMemoryProvider::new($crate::current_function!())) }
    };
}

#[cfg(test)]
use std::cell::RefCell;
#[cfg(test)]
use std::collections::HashSet;

#[cfg(test)]
pub(self) struct TraceDrop {
    content: i32,
    drop_tracer: Rc<RefCell<HashSet<i32>>>
}

#[cfg(test)]
impl Drop for TraceDrop {
        
    fn drop(&mut self) {
        self.drop_tracer.as_ref().borrow_mut().insert(self.content);
    }
}

#[test]
fn test_new_init_drop_after_error() {

    let drop_tracer = Rc::new(RefCell::new(HashSet::new()));

    let result = default_memory_provider!().try_new_init(16, |i| if i < 8 {
        Ok(TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() })
    } else {
        Err(i)
    });
    assert_eq!(8, result.err().unwrap());
    assert_eq!((0..8).collect::<HashSet<_>>(), *drop_tracer.as_ref().borrow());
}

#[test]
fn test_new_init_drop() {

    let drop_tracer = Rc::new(RefCell::new(HashSet::new()));

    {
        default_memory_provider!().new_init(12, |i| TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() });
    }

    assert_eq!((0..12).collect::<HashSet<_>>(), *drop_tracer.as_ref().borrow());
}

#[test]
fn test_new_init() {
    let result = default_memory_provider!().new_init(10, |i| i);
    assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..], &*result);
}

#[test]
fn test_type_erasure() {
    let memory_provider: MemProviderRc<i32> = default_memory_provider!();
    memory_provider.new_init(100, |i| i as i32);
}