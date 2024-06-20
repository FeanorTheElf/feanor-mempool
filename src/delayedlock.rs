use std::{cell::UnsafeCell, marker::PhantomData, ops::Deref, ptr::NonNull, sync::*};
use atomic::AtomicUsize;

use crate::lockfree::LockfreeQueue;

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
    data: UnsafeCell<T>,
    data_lock: RwLock<PhantomData<T>>,
    require_locked: AtomicUsize,
    current_readers: AtomicUsize,
    write_tasks: LockfreeQueue<Box<dyn Send + FnOnce(&mut T)>>
}

impl<T: Send + Sync> ReadDelayedWriteLock<T> {
    
    pub fn new(data: T) -> Self {
        Self {
            data_lock: RwLock::new(PhantomData),
            data: UnsafeCell::new(data),
            require_locked: AtomicUsize::new(0),
            current_readers: AtomicUsize::new(0),
            write_tasks: LockfreeQueue::new()
        }
    }
}

fn atomic_inc_no_overflow(var: &AtomicUsize) {
    let mut current = var.load(atomic::Ordering::SeqCst);
    loop {
        assert!(current != usize::MAX, "overflow in atomic counter, cannot proceed without risking UB");
        match var.compare_exchange_weak(current, current + 1, atomic::Ordering::SeqCst, atomic::Ordering::SeqCst) {
            Ok(_) => { return; },
            Err(new_value) => { current = new_value; }
        }
    }
}

pub struct ReadDelayedWriteLockReadGuard<'a, T: Send + Sync> {
    lock: &'a ReadDelayedWriteLock<T>,
    read_guard: Option<RwLockReadGuard<'a, PhantomData<T>>>
}

pub struct MappedReadDelayedWriteLockReadGuard<'a, T: Send + Sync, U: Send + Sync> {
    data: *const U,
    #[allow(dead_code)]
    lock: ReadDelayedWriteLockReadGuard<'a, T>
}

impl<'a, T: Send + Sync> ReadDelayedWriteLockReadGuard<'a, T> {

    pub fn try_map<U, F: for<'b> FnOnce(&'b T) -> Option<&'b U>>(self, func: F) -> Option<MappedReadDelayedWriteLockReadGuard<'a, T, U>>
        where U: 'a + Send + Sync
    {
        let new_data = func(&*self)? as *const U;
        Some(MappedReadDelayedWriteLockReadGuard {
            lock: self,
            data: new_data
        })
    }
}

impl<'a, T: Send + Sync, U: Send + Sync> Deref for MappedReadDelayedWriteLockReadGuard<'a, T, U> {

    type Target = U;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data }
    }
}

impl<'a, T: Send + Sync> Deref for ReadDelayedWriteLockReadGuard<'a, T> {

    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T: Send + Sync> Drop for ReadDelayedWriteLockReadGuard<'a, T> {

    fn drop(&mut self) {
        let old_current_readers = self.lock.current_readers.fetch_sub(1, atomic::Ordering::SeqCst);
        if old_current_readers == 0 {
            self.lock.try_run_tasks();
        }
        // Explicitly drop `read_guard`; this is not necessary, but removes the "unused" warning
        self.read_guard.take();
    }
}

unsafe impl<T: Send + Sync> Send for ReadDelayedWriteLock<T> {}

unsafe impl<T: Send + Sync> Sync for ReadDelayedWriteLock<T> {}

impl<T: Send + Sync> ReadDelayedWriteLock<T> {

    ///
    /// Returns `true` if it actually managed to acquire a write lock, i.e.
    /// ran any tasks that were enqued when the function was started.
    /// 
    /// Note that if the return value is `true` and a task was enqueued when
    /// the `try_run_tasks()` was called, this function has been run. However,
    /// it might have been run by another thread.
    /// 
    fn try_run_tasks(&self) -> bool {
        atomic_inc_no_overflow(&self.require_locked);
        let mut result = false;
        if self.current_readers.load(atomic::Ordering::SeqCst) == 0 {
            self.run_tasks();
            result = true;
        }
        self.require_locked.fetch_sub(1, atomic::Ordering::SeqCst);
        return result;
    }

    fn run_tasks(&self) {
        assert!(self.require_locked.load(atomic::Ordering::SeqCst) > 0);
        assert!(self.current_readers.load(atomic::Ordering::SeqCst) == 0);
        let write_guard = self.data_lock.write().unwrap_or_else(|e| e.into_inner());
        while let Ok(func) = self.write_tasks.try_dequeue() {
            let data_mut = unsafe { &mut *self.data.get() };
            func(data_mut);
        }
        drop(write_guard)
    }

    pub fn read<'a>(&'a self) -> ReadDelayedWriteLockReadGuard<'a, T> {
        atomic_inc_no_overflow(&self.current_readers);
        if self.require_locked.load(atomic::Ordering::SeqCst) > 0 {
            let read_guard = self.data_lock.read().unwrap_or_else(|e| e.into_inner());
            return ReadDelayedWriteLockReadGuard {
                lock: self,
                read_guard: Some(read_guard)
            };
        } else {
            return ReadDelayedWriteLockReadGuard {
                lock: self,
                read_guard: None
            };
        }
    }

    pub fn query_write(&self, func: Box<dyn Send + FnOnce(&mut T)>) -> bool {
        atomic_inc_no_overflow(&self.require_locked);
        let result = if self.current_readers.load(atomic::Ordering::SeqCst) == 0 {
            let write_guard = self.data_lock.write().unwrap_or_else(|e| e.into_inner());
            let data_mut = unsafe { &mut *self.data.get() };
            func(data_mut);
            drop(write_guard);
            true
        } else {
            self.write_tasks.try_enqueue(func).ok().unwrap();
            self.try_run_tasks()
        };
        self.require_locked.fetch_sub(1, atomic::Ordering::SeqCst);
        return result;
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