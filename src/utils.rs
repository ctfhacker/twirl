//! Utilities

/// Allocate a Read|Write buffer
#[cfg(target_os = "linux")]
pub fn alloc_rw(size: usize) -> *mut u8 {
    extern "C" {
        fn mmap(
            addr: *mut u8,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: usize,
        ) -> *mut u8;
    }

    // const PROT_EXEC: i32 = 4;
    const PROT_READ: i32 = 2;
    const PROT_WRITE: i32 = 1;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MAP_PRIVATE: i32 = 2;

    // Return an RWX Priv/Anon allocated buffer of `size` bytes
    unsafe {
        let res = mmap(
            std::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(!res.is_null(), "Failed to allocate buffer of size {size}");

        res
    }
}

#[cfg(not(target_os = "linux"))]
compile_error!("Allocation not implemented for non-linux OS");
