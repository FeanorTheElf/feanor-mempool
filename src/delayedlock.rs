use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::*;
use atomic::AtomicI64;

use crate::lockfree::LockfreeQueue;

type PhantomUnsend = PhantomData<MutexGuard<'static, ()>>;

fn atomic_cas_loop<F: FnMut(i64) -> Result<i64, E>, E>(var: &AtomicI64, mut func: F) -> Result<i64, E> {
    let mut current = var.load(atomic::Ordering::SeqCst);
    loop {
        match func(current) {
            Ok(new_val) => match var.compare_exchange_weak(current, new_val, atomic::Ordering::SeqCst, atomic::Ordering::SeqCst) {
                Ok(old_value) => { return Ok(old_value); },
                Err(new_value) => { current = new_value; }
            },
            Err(e) => { return Err(e); }
        }
    }
}

///
/// An optimistic Read-Async-Lock.
/// 
/// As opposed to a read-write-lock, this lock can either be read-locked multiple times (meaning
/// multiple threads read data), or async-locked multiple times (meaning multiple threads access
/// the locked data using atomic operation). Note that it is UB if some threads normally read data,
/// while other threads atomically operate on it, so this is prevented by the lock.
/// 
struct TryRALock {
    // negative -> async-locked, positive -> read-locked
    locks: AtomicI64
}

struct TryRALockReadLockGuard<'a> {
    parent: &'a TryRALock,
    decrement_on_drop: bool,
    _dont_send: PhantomUnsend
}

impl<'a> TryRALockReadLockGuard<'a> {

    pub fn release_try_async(mut self) -> Option<TryRALockAsyncLockGuard<'a>> {
        match self.parent.locks.compare_exchange(1, -1, atomic::Ordering::SeqCst, atomic::Ordering::SeqCst) {
            Ok(_) => {
                self.decrement_on_drop = false;
                Some(TryRALockAsyncLockGuard { parent: self.parent, _dont_send: PhantomData })
            },
            Err(_) => None
        }
    }
}

impl<'a> Drop for TryRALockReadLockGuard<'a> {
    fn drop(&mut self) {
        if self.decrement_on_drop {
            let old_val = self.parent.locks.fetch_sub(1, atomic::Ordering::SeqCst);
            debug_assert!(old_val > 0);
        }
    }
}

struct TryRALockAsyncLockGuard<'a> {
    parent: &'a TryRALock,
    _dont_send: PhantomUnsend
}

impl<'a> Drop for TryRALockAsyncLockGuard<'a> {
    fn drop(&mut self) {
        let old_val = self.parent.locks.fetch_add(1, atomic::Ordering::SeqCst);
        debug_assert!(old_val < 0);
    }
}

impl TryRALock {

    fn new() -> Self {
        Self { locks: AtomicI64::new(0) }
    }

    fn async_lock<'a>(&'a self) -> Option<TryRALockAsyncLockGuard<'a>> {
        match atomic_cas_loop(&self.locks, |locked| if locked <= 0 { Ok(locked.checked_sub(1).unwrap()) } else { Err(()) }) {
            Ok(_) => Some(TryRALockAsyncLockGuard { parent: self, _dont_send: PhantomData }),
            Err(()) => None
        }
    }

    fn read_or_async<'a>(&'a self) -> Result<TryRALockReadLockGuard<'a>, TryRALockAsyncLockGuard<'a>> {
        let prev = atomic_cas_loop::<_, !>(&self.locks, |locked| if locked >= 0 { Ok(locked.checked_add(1).unwrap()) } else { Ok(locked.checked_sub(1).unwrap()) }).unwrap_or_else(|x| x);
        if prev >= 0 {
            Ok(TryRALockReadLockGuard { parent: self, decrement_on_drop: true, _dont_send: PhantomData })
        } else {
            Err(TryRALockAsyncLockGuard { parent: self, _dont_send: PhantomData })
        }
    }
}

///
/// A read-write lock that tries to "lock" the data on reads in a lock-free way,
/// whenever possible.
/// 
/// This however means that when trying to get a write lock, it might be impossible
/// to wait until all reads are done. Instead, the lock will accept the write function,
/// but delay its execution until no thread is currently reading the data.
/// 
/// During the execution of a function having a write lock, all further reads will
/// also try to get a read lock, and thus block until the write function is done.
/// 
/// # Behavior on panics
/// 
/// If an accepted write function panics, this is safe (i.e. not UB), but will cause
/// the `ReadDelayedWriteLock` to possibly malfunction from then on. In particular, it
/// might from then on always acquire read and write locks, and never work in a lock-free
/// way anymore.
/// 
pub struct ReadDelayedWriteLock<T: Send + Sync> {
    actual_data: UnsafeCell<T>,
    // to read, need either an optimistic read lock, or an optimistic async lock + a fallback read lock;
    // to write, need an optimistic async lock + a fallback write lock
    optimistic_lock: TryRALock,
    fallback_lock: RwLock<PhantomData<T>>,
    delayed_writes: LockfreeQueue<Box<dyn Send + FnOnce(&mut T)>>
}

impl<T: Send + Sync> ReadDelayedWriteLock<T> {
    
    pub fn new(data: T) -> Self {
        Self {
            optimistic_lock: TryRALock::new(),
            fallback_lock: RwLock::new(PhantomData),
            actual_data: UnsafeCell::new(data),
            delayed_writes: LockfreeQueue::new()
        }
    }
}

unsafe impl<T: Send + Sync> Send for ReadDelayedWriteLock<T> {}
unsafe impl<T: Send + Sync> Sync for ReadDelayedWriteLock<T> {}
    
enum ReadDelayedWriteLockReadLockGuardCore<'a, T: Send + Sync> {
    Optimistic(Option<TryRALockReadLockGuard<'a>>, &'a ReadDelayedWriteLock<T>),
    Fallback(TryRALockAsyncLockGuard<'a>, Option<RwLockReadGuard<'a, PhantomData<T>>>, &'a ReadDelayedWriteLock<T>)
}

pub struct ReadDelayedWriteLockReadLockGuard<'a, T: Send + Sync> {
    data: ReadDelayedWriteLockReadLockGuardCore<'a, T>
}

impl<'a, T: Send + Sync> ReadDelayedWriteLockReadLockGuard<'a, T> {

    pub fn try_map<F: for<'b> FnOnce(&'b T) -> Result<&'b U, E>, U: 'a + Send + Sync, E>(self, func: F) -> Result<MappedReadDelayedWriteLockReadLockGuard<'a, T, U>, E> {
        let ptr = func(&self).map(|ptr| ptr as *const U);
        ptr.map(|ptr| MappedReadDelayedWriteLockReadLockGuard { actual_lock: self, data_ptr: ptr as *const U })
    }
}

pub struct MappedReadDelayedWriteLockReadLockGuard<'a, T: Send + Sync, U: Send + Sync> {
    actual_lock: ReadDelayedWriteLockReadLockGuard<'a, T>,
    data_ptr: *const U
}

impl<'a, T: Send + Sync, U: Send + Sync> Deref for MappedReadDelayedWriteLockReadLockGuard<'a, T, U> {

    type Target = U;

    fn deref(&self) -> &Self::Target {
        // ensure that parent lock is actually safe to derefence - should not be necessary, but let's be paranoid
        _ = &*self.actual_lock;
        // this is now safe, since we have the lock
        unsafe { &*self.data_ptr }
    }
}

impl<'a, T: Send + Sync> Deref for ReadDelayedWriteLockReadLockGuard<'a, T> {

    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.data {
            ReadDelayedWriteLockReadLockGuardCore::Optimistic(read_lock, parent) => {
                assert!(read_lock.is_some());
                // this is now safe, since we have a valid read-lock combination
                unsafe { &*parent.actual_data.get() }
            },
            ReadDelayedWriteLockReadLockGuardCore::Fallback(_async_lock, read_lock, parent) => {
                assert!(read_lock.is_some());
                // this is now safe, since we have a valid read-lock combination
                unsafe { &*parent.actual_data.get() }
            }
        }
    }
}

impl<'a, T: Send + Sync> Drop for ReadDelayedWriteLockReadLockGuard<'a, T> {

    fn drop(&mut self) {
        match &mut self.data {
            ReadDelayedWriteLockReadLockGuardCore::Optimistic(read_lock, parent) => if let Some(async_lock) = read_lock.take().unwrap().release_try_async() {
                parent.try_execute_writes(&async_lock)
            },
            ReadDelayedWriteLockReadLockGuardCore::Fallback(async_lock, read_lock, parent) => {
                drop(read_lock.take().unwrap());
                parent.try_execute_writes(async_lock);
            }
        }
    }
}

impl<T: Send + Sync> ReadDelayedWriteLock<T> {

    fn try_execute_writes(&self, _async_lock: &TryRALockAsyncLockGuard) {
        if let Some(write_task) = self.delayed_writes.try_dequeue().ok() {
            let write_guard = self.fallback_lock.write().unwrap_or_else(|poison| poison.into_inner());
            write_task(unsafe { &mut *self.actual_data.get() });
            while let Some(write_task) = self.delayed_writes.try_dequeue().ok() {
                write_task(unsafe { &mut *self.actual_data.get() });
            }
            drop(write_guard);
        }
    }

    pub fn read<'a>(&'a self) -> ReadDelayedWriteLockReadLockGuard<'a, T> {
        ReadDelayedWriteLockReadLockGuard {
            data: match self.optimistic_lock.read_or_async() {
                Ok(read_lock) => ReadDelayedWriteLockReadLockGuardCore::Optimistic(Some(read_lock), self),
                Err(async_lock) => ReadDelayedWriteLockReadLockGuardCore::Fallback(async_lock, Some(self.fallback_lock.read().unwrap_or_else(|poison| poison.into_inner())), self)
            }
        }
    }

    pub fn query_write<F: 'static + Send + FnOnce(&mut T)>(&self, func: F) {
        if let Some(async_lock) = self.optimistic_lock.async_lock() {
            let write_lock = self.fallback_lock.write().unwrap_or_else(|poison| poison.into_inner());
            func(unsafe { &mut *self.actual_data.get() });
            drop(write_lock);
            drop(async_lock);
        } else {
            self.delayed_writes.try_enqueue(Box::new(func)).ok().unwrap();
            if let Some(async_lock) = self.optimistic_lock.async_lock() {
                self.try_execute_writes(&async_lock);
            }
        }
    }
}

pub struct SendableNonNull<T: ?Sized> {
    data: NonNull<T>
}

impl<T: ?Sized> SendableNonNull<T> {

    pub unsafe fn new(data: NonNull<T>) -> Self {
        Self { data }
    }

    pub fn extract(self) -> NonNull<T> {
        self.data
    }
}

unsafe impl<T: ?Sized> Send for SendableNonNull<T> {}