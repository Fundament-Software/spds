#![allow(dead_code)]

use bitfield_struct::bitfield;
use eyre::Result;
#[cfg(not(miri))]
use memmap2::MmapMut;
use std::cell::RefCell;
use std::cell::RefMut;
use std::collections::HashSet;
use std::fs::File;
#[cfg(not(miri))]
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::mem;
use std::mem::size_of;
use std::path::Path;
use std::rc::Rc;

const MAX_CHILDREN: usize = 32;
const MAX_NODE_SIZE: usize = MAX_CHILDREN + 1;
const MIN_NODE_SIZE: usize = 2; // It's very easy to forget that a node must contain at least 1 value
const MAX_NODE_BYTES: usize = size_of::<[u64; MAX_NODE_SIZE]>();
// Note that page size is not always 4096 but this works nicely as the chunk size for flushing
const PAGE_SIZE: usize = 4096;
const FLUSH_THRESHOLD: usize = 2048;

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum Error {
    #[error("Expected memory mapped size of at least {0} but found {1}")]
    MemMapTooSmall(usize, usize),
    #[error("Memory alignment should be aligned to {0} but is off by {1}")]
    InvalidAlignment(usize, usize),
    #[error("Expected file size of at least {0} but found {1}")]
    FileTooSmall(usize, u64),
    #[error("Invalid Trie state! Was this trie not cleanly flushed? (Try using restore)")]
    DirtyTrieState,
    #[error(
        "Requested key is not available (key bits {key_bits:?} at offset {key_offset:?} not found in node {node:?}"
    )]
    NotFound { key_offset: u8, key_bits: u8, node: u64 },
    #[error("Tried to insert key that already exists and has value {0}")]
    AlreadyExists(u64),
    #[error("Ran out of space in backing storage! Storage size: {0}")]
    OutOfMemory(usize),
    #[error("unknown HashedArrayTrie error")]
    Unknown,
}

#[bitfield(u64)]
struct Flags {
    #[bits(32)]
    mask: u32, // Each bit represents a child that exists, and the total number of children is the number of set bits
    #[bits(1)]
    leaf: bool,
    #[bits(8)]
    refcount: u8,
    #[bits(3)]
    skips: u8, // Stores how many skip levels (0-4) we're using in this node
    #[bits(20)]
    skipbits: u32, // stores the skipped bits of the key
}

impl Flags {
    pub fn offset(&self, index: u8) -> usize {
        (((self.0 & 0xFFFFFFFF) << (32 - index)) as u32).count_ones() as usize
    }
    pub fn count(&self) -> usize {
        (self.mask()).count_ones() as usize
    }
    pub fn exists(&self, index: u8) -> bool {
        (self.mask() & (0b1 << index)) != 0
    }
    pub fn append(&mut self, index: u8) {
        assert!(!self.exists(index));
        self.set_mask(self.mask() | (0b1 << index));
    }
    pub fn from_ref_mut(target: &mut u64) -> &mut Flags {
        unsafe { &mut *(target as *mut u64).cast::<Flags>() }
    }
    pub fn from_ref(target: &u64) -> &Flags {
        unsafe { &*(target as *const u64).cast::<Flags>() }
    }

    #[inline]
    fn parity_location(&self) -> u8 {
        // We hide the parity either in the skip count or in the skip bits depending on if skip count is 4 or not
        if (self.0 & (1 << 43)) != 0 {
            41
        } else {
            63
        }
    }
    #[inline]
    fn calc_parity(&self) -> u32 {
        (self.0 & !(1 << self.parity_location())).count_ones() & 1
    }
    #[inline]
    fn get_parity(&self) -> bool {
        (self.0 & (1 << self.parity_location())) != 0
    }
    #[inline]
    pub fn check_parity(&self) -> bool {
        self.get_parity() == (self.calc_parity() != 0)
    }
    #[inline(never)]
    pub fn set_parity(&mut self) {
        let bit = 1 << self.parity_location();
        let parity = (self.calc_parity() as u64) << self.parity_location();
        self.0 = (self.0 & !bit) | parity;
    }
    pub fn is_valid(&self) -> bool {
        // refcount should be greater than zero
        self.refcount() > 0 &&
        // parity bit should be valid
        self.check_parity() &&
        // should have a nonzero number of children (otherwise it's degenerate)
        self.count() > 0 &&
        // skipbits should be zero for all skiplevels we aren't using
        if (self.skips() & 0b100) != 0 {
            (self.skips() & 0b010) == 0
        } else {
            (self.skipbits() & !(1 << 19) & !((1 << (self.skips() * 5)) - 1)) == 0
        }
    }
}

#[test]
fn test_offsets() {
    let mut a = Flags::new().with_refcount(1);

    assert_eq!(a.count(), 0);
    assert!(!a.exists(0));
    assert!(!a.exists(1));
    assert!(!a.exists(31));
    assert_eq!(a.offset(0), 0);
    assert_eq!(a.offset(1), 0);
    assert_eq!(a.offset(31), 0);

    a.append(1);
    assert_eq!(a.count(), 1);
    assert!(!a.exists(0));
    assert!(a.exists(1));
    assert!(!a.exists(2));
    assert!(!a.exists(31));
    assert_eq!(a.offset(0), 0);
    assert_eq!(a.offset(1), 0);
    assert_eq!(a.offset(2), 1);
    assert_eq!(a.offset(31), 1);

    a.append(0);
    assert_eq!(a.count(), 2);
    assert!(a.exists(0));
    assert!(a.exists(1));
    assert!(!a.exists(2));
    assert!(!a.exists(31));
    assert_eq!(a.offset(0), 0);
    assert_eq!(a.offset(1), 1);
    assert_eq!(a.offset(2), 2);
    assert_eq!(a.offset(31), 2);

    a.append(31);
    assert_eq!(a.count(), 3);
    assert!(a.exists(0));
    assert!(a.exists(1));
    assert!(!a.exists(2));
    assert!(a.exists(31));
    assert_eq!(a.offset(0), 0);
    assert_eq!(a.offset(1), 1);
    assert_eq!(a.offset(2), 2);
    assert_eq!(a.offset(31), 2);

    a.append(2);
    assert_eq!(a.count(), 4);
    assert!(a.exists(0));
    assert!(a.exists(1));
    assert!(a.exists(2));
    assert!(a.exists(31));
    assert_eq!(a.offset(0), 0);
    assert_eq!(a.offset(1), 1);
    assert_eq!(a.offset(2), 2);
    assert_eq!(a.offset(31), 3);

    for i in 3..=30 {
        a.append(i);
    }

    assert_eq!(a.count(), 32);

    for i in 0..=31 {
        assert!(a.exists(i));
        assert_eq!(a.offset(i), i.into());
    }
}

#[test]
fn test_parity() {
    let mut a = Flags::new().with_refcount(0);

    assert_eq!(a.0, 0);
    assert!(!a.get_parity());
    assert_eq!(a.calc_parity(), 0);
    assert!(a.check_parity());
    a.set_parity();
    assert_eq!(a.0, 0);
    assert!(!a.get_parity());
    assert_eq!(a.calc_parity(), 0);
    assert!(a.check_parity());

    a.set_refcount(1);
    assert_ne!(a.0, 0);
    assert!(!a.get_parity());
    assert_eq!(a.calc_parity(), 1);
    assert!(!a.check_parity());
    a.set_parity();
    assert!(a.get_parity());
    assert_eq!(a.calc_parity(), 1);
    assert!(a.check_parity());

    a.0 = 0;
    assert!(!a.get_parity());
    a.set_skips(1);
    assert!(!a.get_parity());
    a.set_skips(4);
    assert!(!a.get_parity());
    a.set_skips(5);
    assert!(a.get_parity());
    a.set_skips(6);
    assert!(!a.get_parity());
    a.set_skips(7);
    assert!(a.get_parity());
    a.set_skips(1);
    assert!(!a.get_parity());
    a.set_skipbits(1 << 19);
    assert!(a.get_parity());
    a.set_skipbits(1 << 18);
    assert!(!a.get_parity());
}

#[derive(Debug)]
#[repr(C)]
struct Header {
    freelist: [u64; 32], // Maintains a freelist for all 32 non-leaf node sizes
    root: u64,           // This is the primary root
    clean: u64,          // Only a 1 bit value but set to a u64 to ensure alignment
}

#[derive(Debug)]
pub struct Storage {
    handle: Option<File>,
    #[cfg(miri)]
    mapping: Option<Vec<u8>>,
    #[cfg(not(miri))]
    mapping: Option<MmapMut>,
    // TODO: Maybe replace with data structure that can maintain efficient merged intervals
    //dirty_pages: HashSet<usize>,
}

const HEADER_BYTES: usize = size_of::<Header>();

impl Storage {
    #[inline]
    fn parts(&self) -> (&Header, &[u64]) {
        unsafe {
            let (header, slice) = self.mapping.as_ref().unwrap_unchecked().split_at(HEADER_BYTES);
            (
                &*(header.as_ptr() as *const Header),
                std::slice::from_raw_parts(slice.as_ptr() as *const u64, slice.len() / size_of::<u64>()),
            )
        }
    }

    #[inline]
    fn parts_mut(&mut self) -> (&mut Header, &mut [u64]) {
        unsafe {
            let (header, slice) = self.mapping.as_mut().unwrap_unchecked().split_at_mut(HEADER_BYTES);
            (
                &mut *(header.as_mut_ptr() as *mut Header),
                std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u64, slice.len() / size_of::<u64>()),
            )
        }
    }

    #[inline]
    fn words(&self) -> &[u64] {
        unsafe {
            let slice = &self.mapping.as_ref().unwrap_unchecked()[HEADER_BYTES..];
            std::slice::from_raw_parts(slice.as_ptr() as *const u64, slice.len() / size_of::<u64>())
        }
    }

    #[inline]
    fn words_mut(&mut self) -> &mut [u64] {
        unsafe {
            let slice = &mut self.mapping.as_mut().unwrap_unchecked()[HEADER_BYTES..];
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u64, slice.len() / size_of::<u64>())
        }
    }

    unsafe fn init_self(&mut self) -> Result<()> {
        // We do this alignment check once, and the rest of the time we do the unguarded direct mutation.
        let (prefix, slice, tail) = &mut self.mapping.as_mut().unwrap_unchecked()[HEADER_BYTES..].align_to_mut::<u64>();

        if !prefix.is_empty() {
            return Err(Error::InvalidAlignment(size_of::<u64>(), prefix.len()).into());
        } else if !tail.is_empty() {
            return Err(Error::InvalidAlignment(size_of::<u64>(), tail.len()).into());
        }

        // Initialize our canary integer at offset 0, which is always invalid.
        slice[0] = u64::MAX;
        // Initialize the root node as empty
        slice[1] = Flags::new().with_refcount(1).into();

        {
            let (header, _) = self.parts_mut();

            // Initialize freelist and dirty clean value
            header.freelist.fill(u64::MAX);
            header.clean = 0;
            header.root = 1;
        }

        // If init section fails, we will leave a file full of zeros, which is okay because that's considered invalid
        // anyway.
        self.init_section(
            HEADER_BYTES + MAX_NODE_BYTES + size_of::<u64>(),
            self.mapping.as_ref().unwrap_unchecked().len(),
        )?;

        // flush our clean value of 0 so we can detect if we aren't closed properly
        #[cfg(not(miri))]
        self.mapping.as_mut().unwrap_unchecked().flush_range(0, HEADER_BYTES)?;

        Ok(())
    }

    #[cfg(not(miri))]
    unsafe fn new_inner(handle: Option<File>, mapping: MmapMut) -> Result<Storage> {
        if mapping.len() < HEADER_BYTES {
            return Err(Error::MemMapTooSmall(HEADER_BYTES, mapping.len()).into());
        }

        let mut result = Storage {
            handle,
            mapping: Some(mapping),
            //dirty_pages: HashSet::new(),
        };

        result.init_self()?;
        Ok(result)
    }

    #[cfg(not(miri))]
    pub fn new_file(src: File, new_size: u64) -> Result<Storage> {
        assert_eq!(
            new_size % size_of::<u64>() as u64,
            0,
            "size MUST be a multiple of an unsigned 64-bit integer"
        );

        // Create a new file with enough room for the header, a canary word, and then one maximum size node
        src.set_len((HEADER_BYTES + size_of::<u64>() + MAX_NODE_BYTES) as u64 + new_size)?;
        unsafe {
            let mapping = MmapMut::map_mut(&src)?;
            Self::new_inner(Some(src), mapping)
        }
    }

    #[cfg(not(miri))]
    pub fn new_ref(src: &File, new_size: u64) -> Result<Storage> {
        assert_eq!(
            new_size % size_of::<u64>() as u64,
            0,
            "size MUST be a multiple of an unsigned 64-bit integer"
        );

        // Create a new file with enough room for the header, a canary word, and then one maximum size node
        src.set_len((HEADER_BYTES + size_of::<u64>() + MAX_NODE_BYTES) as u64 + new_size)?;
        unsafe { Self::new_inner(None, MmapMut::map_mut(src)?) }
    }

    #[cfg(not(miri))]
    pub fn new(path: &Path, new_size: u64) -> Result<Storage> {
        let src = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        Self::new_file(src, new_size)
    }

    #[cfg(miri)]
    pub fn new(_: &Path, new_size: u64) -> Result<Storage> {
        unsafe {
            let mut aligned = Vec::<u64>::new();
            aligned.resize(
                (HEADER_BYTES + size_of::<u64>() + MAX_NODE_BYTES + new_size as usize) / size_of::<u64>(),
                0,
            );
            let alignlen = aligned.len();
            let aligncap = aligned.capacity();
            let mapping = Vec::from_raw_parts(
                aligned.leak().as_mut_ptr() as *mut u8,
                alignlen * size_of::<u64>(),
                aligncap * size_of::<u64>(),
            );

            let mut result = Storage {
                handle: None,
                mapping: Some(mapping),
                //dirty_pages: HashSet::new(),
            };

            result.init_self()?;
            return Ok(result);
        }
    }

    pub fn allocate(&mut self, count: usize) -> Result<u64> {
        assert!(count > 0);
        assert!(count <= 32);
        if self.mapping.is_some() {
            let (header, words) = self.parts_mut();
            let freelist = &mut header.freelist;
            for i in (count - 1)..MAX_CHILDREN {
                if freelist[i] != u64::MAX {
                    let result = freelist[i];
                    freelist[i] = words[freelist[i] as usize];
                    if freelist[i] != u64::MAX {
                        assert!(freelist[i] as usize + i + 1 < words.len());
                    }

                    // If we have more than 1 extra space left over, assign it to different freelist instead of wasting
                    // it
                    if count < i {
                        // i is in terms of the index, which is the true size - 2 (MIN_NODE_SIZE), whereas count is the
                        // true size - 1
                        let remainder = (i + 2) - (count + 1);
                        // We express remainder in the true word size because we need at least 2 words to squeeze in 1
                        // node (1 header + 1 data)
                        if remainder >= MIN_NODE_SIZE {
                            // However, now that remainder is in the true word size, we have to subtract MIN_NODE_SIZE
                            // to get to the actual freelist index!
                            let split_node = result as usize + count + 1;
                            words[split_node] = freelist[remainder - MIN_NODE_SIZE];
                            freelist[remainder - MIN_NODE_SIZE] = split_node as u64;
                        }
                    }
                    return Ok(result);
                }
            }
            Err(Error::OutOfMemory(self.mapping.as_ref().unwrap().len()).into())
        } else {
            Err(Error::OutOfMemory(0).into())
        }
    }

    // Initializes a new section at the given byte offset by adding it to the freelist. We do this in reverse order so
    // the freelist doesn't try to fill up the new section backwards.
    fn init_section(&mut self, byte_offset: usize, byte_end: usize) -> Result<()> {
        if let Some(mapping) = self.mapping.as_mut() {
            unsafe {
                let ptr = mapping.as_mut_ptr();
                let unaligned: &mut [u8] = &mut mapping[byte_offset..byte_end];
                let header = &mut *(ptr as *mut Header);
                let (prefix, slice, _) = unaligned.align_to_mut::<u64>();
                if !prefix.is_empty() {
                    // If this happens, then there is a high chance that self.mapping itself is not u64 aligned, which
                    // is very bad.
                    return Err(Error::InvalidAlignment(size_of::<u64>(), prefix.len()).into());
                }

                // First we get the head and add it to the proper freelist, if there is one
                let mut word_offset = (byte_offset - HEADER_BYTES) / size_of::<u64>();
                let headsize = slice.len() % MAX_NODE_SIZE;
                let node_aligned = if headsize >= MIN_NODE_SIZE {
                    slice[0] = header.freelist[headsize - MIN_NODE_SIZE];
                    header.freelist[headsize - MIN_NODE_SIZE] = word_offset as u64;
                    word_offset += headsize;
                    &mut slice[headsize..]
                } else {
                    let count = slice.len();
                    // 1 extra u64 isn't big enough to put in our freelists so we just ignore it
                    &mut slice[..count - headsize]
                };

                assert_eq!(node_aligned.len() % MAX_NODE_SIZE, 0);

                // Then we add the remaining max sized nodes to the freelist in reverse
                for i in (0..node_aligned.len()).step_by(MAX_NODE_SIZE).rev() {
                    node_aligned[i] = header.freelist[MAX_NODE_SIZE - MIN_NODE_SIZE];
                    header.freelist[MAX_NODE_SIZE - MIN_NODE_SIZE] = (word_offset + i) as u64;
                }
            }
            Ok(())
        } else {
            Err(Error::OutOfMemory(0).into())
        }
    }

    #[cfg(not(miri))]
    pub fn load_file(src: File) -> Result<Storage> {
        let fsize = src.metadata()?.len();
        if fsize < HEADER_BYTES as u64 {
            return Err(Error::FileTooSmall(HEADER_BYTES, fsize).into());
        }

        unsafe {
            let mapping = MmapMut::map_mut(&src)?;
            if mapping.len() < HEADER_BYTES {
                return Err(Error::MemMapTooSmall(HEADER_BYTES, mapping.len()).into());
            }
            let mut storage = Storage {
                mapping: Some(mapping),
                handle: Some(src),
                //dirty_pages: HashSet::new(),
            };

            let slice: &[u8] = &storage.mapping.as_ref().unwrap_unchecked()[..HEADER_BYTES];
            let prefix = slice.align_to::<u64>().0;
            if !prefix.is_empty() {
                return Err(Error::InvalidAlignment(size_of::<u64>(), prefix.len()).into());
            }
            let (header, _) = storage.parts_mut();
            if header.clean == 0 {
                return Err(Error::DirtyTrieState.into());
            }

            // Set our clean value to 0 and flush so we can detect if we aren't closed properly
            header.clean = 0;

            storage
                .mapping
                .as_mut()
                .unwrap_unchecked()
                .flush_range(0, HEADER_BYTES)?;
            Ok(storage)
        }
    }

    #[cfg(not(miri))]
    // Load a file - if it doesn't already exist or is invalid, returns an error
    pub fn load(path: &Path) -> Result<Storage> {
        Self::load_file(OpenOptions::new().read(true).write(true).create(false).open(path)?)
    }

    fn scan_valid_nodes(offset: u64, words: &mut [u64], valid: &mut HashSet<u64>) -> bool {
        let total_len = words.len() as u64;
        if offset >= total_len {
            return false;
        }

        let offset = offset as usize;
        // Ensure this node is consistent.
        let node = Flags::from_ref_mut(&mut words[offset]);

        if !node.is_valid() {
            return false;
        }

        if !node.leaf() {
            // Check children for consistency. Remove any that have been corrupted.
            let mut count = 0;
            let mut valid_children = 0;
            let mut children = [0; 32];
            let mut mask = node.mask();
            for i in 0..32 {
                if (mask & (1 << i)) != 0 {
                    if Self::scan_valid_nodes(words[offset + 1 + count], words, valid) {
                        children[valid_children] = words[offset + 1 + count];
                        valid_children += 1;
                    } else {
                        mask &= !(1 << i);
                    }
                    count += 1;
                }
            }

            if valid_children == 0 {
                return false;
            }

            // If we had some invalid children, we reconstruct the node with the valid ones
            if valid_children != count {
                words[offset + 1..(offset + 1 + count)].fill(0);
                words[offset + 1..(offset + 1 + valid_children)].copy_from_slice(&children[..valid_children]);
            }

            assert_eq!(valid_children as u32, mask.count_ones());

            let node = Flags::from_ref_mut(&mut words[offset]);
            node.set_mask(mask);
            node.set_parity();

            for i in 0..=node.count() {
                valid.insert((offset + i) as u64);
            }
        } else if offset as u64 + node.count() as u64 >= total_len {
            return false;
        } else {
            for i in 0..=node.count() {
                valid.insert((offset + i) as u64);
            }
        }

        true
    }

    unsafe fn restore_inner(storage: &mut Storage) -> Result<()> {
        let slice: &[u8] = &storage.mapping.as_ref().unwrap_unchecked()[..HEADER_BYTES];
        let prefix = slice.align_to::<u64>().0;
        if !prefix.is_empty() {
            return Err(Error::InvalidAlignment(size_of::<u64>(), prefix.len()).into());
        }

        // Clean the header out, setting root to 1 if it's invalid and wiping the freelist.
        let (header, words) = storage.parts_mut();
        header.clean = 0;
        if header.root == 0 || header.root as usize > words.len() {
            header.root = 1;
        }
        header.freelist.fill(u64::MAX);

        let mut validnodes = HashSet::new();
        Self::scan_valid_nodes(header.root, words, &mut validnodes);

        words[0] = u64::MAX;
        let mut count = 0;
        // Now we reconstruct the freelists by scanning the entire file in reverse.
        for i in (34..words.len() as u64).rev() {
            let valid = validnodes.contains(&i);
            let target = if !valid {
                count += 1;
                i
            } else {
                i + 1
            };

            if valid || count >= MAX_NODE_SIZE {
                assert!(count <= MAX_NODE_SIZE);
                if count >= MIN_NODE_SIZE {
                    words[target as usize] = header.freelist[count - MIN_NODE_SIZE];
                    header.freelist[count - MIN_NODE_SIZE] = target;
                }

                count = 0;
            }
        }

        Ok(())
    }

    #[cfg(not(miri))]
    // Attempts to reconstruct a file that was not cleanly flushed.
    pub fn restore_file(src: File) -> Result<Storage> {
        // We create a hashmap of all reachable, valid nodes starting from the root. Because all our freelists have at
        // least one 0 following them, no valid node should ever have zero children, and no legitimate offset
        // should ever point at 0, so an invalid section is anything pointing to 0, or pointing to a non-leaf
        // node followed by a zero.

        let fsize = src.metadata()?.len();
        if fsize < HEADER_BYTES as u64 {
            return Err(Error::FileTooSmall(HEADER_BYTES, fsize).into());
        }

        unsafe {
            let mapping = MmapMut::map_mut(&src)?;
            if mapping.len() < HEADER_BYTES {
                return Err(Error::MemMapTooSmall(HEADER_BYTES, mapping.len()).into());
            }
            let mut storage = Storage {
                mapping: Some(mapping),
                handle: Some(src),
                //dirty_pages: HashSet::new(),
            };

            Self::restore_inner(&mut storage)?;
            // Now that we've recovered the file, flush the whole thing, keeping clean at 0 since we haven't closed it
            // yet.
            storage.mapping.as_mut().unwrap_unchecked().flush()?;
            Ok(storage)
        }
    }

    #[cfg(not(miri))]
    pub fn restore(path: &Path) -> Result<Storage> {
        Self::restore_file(OpenOptions::new().read(true).write(true).create(false).open(path)?)
    }

    #[cfg(miri)]
    pub fn restore(src: Vec<u8>) -> Result<Storage> {
        unsafe {
            if src.len() < HEADER_BYTES {
                return Err(Error::MemMapTooSmall(HEADER_BYTES, src.len()).into());
            }
            let mut storage = Storage {
                mapping: Some(src),
                handle: None,
                //dirty_pages: HashSet::new(),
            };

            Self::restore_inner(&mut storage)?;
            Ok(storage)
        }
    }

    #[cfg(miri)]
    pub fn resize(&mut self) -> Result<()> {
        if let Some(mapref) = &mut self.mapping {
            unsafe {
                let mapping = std::mem::take(mapref);
                let maplen = mapping.len();
                let mapcap = mapping.capacity();
                let mut aligned = Vec::<u64>::from_raw_parts(
                    mapping.leak().as_mut_ptr() as *mut u64,
                    maplen / size_of::<u64>(),
                    mapcap / size_of::<u64>(),
                );
                aligned.resize(aligned.len() * 2, 0);
                let alignlen = aligned.len();
                let aligncap = aligned.capacity();
                let mut replace = Vec::from_raw_parts(
                    aligned.leak().as_mut_ptr() as *mut u8,
                    alignlen * size_of::<u64>(),
                    aligncap * size_of::<u64>(),
                );
                mem::swap(mapref, &mut replace);

                self.init_section(maplen, self.mapping.as_ref().unwrap_unchecked().len())?;
            }
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[cfg(not(miri))]
    pub fn resize(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(handle) = self.handle.as_ref() {
            let old = mem::take(&mut self.mapping);
            if let Some(mut m) = old {
                let old_size = m.len();
                let new_size = old_size * 2;
                // Attempt an in-place resize
                unsafe {
                    self.mapping = Some(if handle.set_len(new_size as u64).is_ok() {
                        if m.remap(new_size, memmap2::RemapOptions::new().may_move(true)).is_err() {
                            drop(m);
                            MmapMut::map_mut(handle)?
                        } else {
                            m
                        }
                    } else {
                        drop(m);
                        handle.set_len(new_size as u64)?;
                        MmapMut::map_mut(handle)?
                    });

                    self.init_section(old_size, self.mapping.as_ref().unwrap_unchecked().len())?;
                }
                return Ok(());
            }
        }

        Err(Error::OutOfMemory(0).into())
    }

    #[cfg(not(target_os = "linux"))]
    #[cfg(not(miri))]
    pub fn resize(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(handle) = self.handle.as_ref() {
            let old = mem::take(&mut self.mapping);
            if let Some(m) = old {
                let old_size = m.len();
                let new_size = old_size * 2;
                drop(m);
                handle.set_len(new_size as u64)?;

                unsafe {
                    self.mapping = Some(MmapMut::map_mut(handle)?);
                    self.init_section(old_size, self.mapping.as_ref().unwrap_unchecked().len())?;
                }
                return Ok(());
            }
        }

        Err(Error::OutOfMemory(0).into())
    }

    pub fn flush(&mut self) -> Result<()> {
        #[cfg(not(miri))]
        if let Some(mapping) = self.mapping.as_mut() {
            mapping.flush()?;
        }
        Ok(())
    }

    pub fn flush_async(&mut self) -> Result<()> {
        #[cfg(not(miri))]
        if let Some(mapping) = self.mapping.as_mut() {
            mapping.flush_async()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct HashedArrayTrie<K>
where
    K: num::PrimInt,
{
    pub storage: Rc<RefCell<Storage>>,
    offset: u64,
    phantomkey: PhantomData<K>,
}

impl<K> HashedArrayTrie<K>
where
    K: num::PrimInt + num::cast::AsPrimitive<u8>,
{
    pub fn new(storage: &Rc<RefCell<Storage>>, root: u64) -> HashedArrayTrie<K> {
        HashedArrayTrie {
            storage: Rc::clone(storage),
            offset: root,
            phantomkey: PhantomData,
        }
    }

    // DOES NOT INCREMENT REFCOUNTS. This is to enable the mutable fast-path, which will immediately delete the source
    // node.
    #[inline]
    fn clone_node(from: usize, to: u64, count: usize, words: &mut [u64]) {
        // TODO: this uses memmove, these should never overlap so a copy is slightly faster but it might not matter for
        // small sizes
        words.copy_within(from..(from + count + 1), to as usize)
    }

    // This fixes the refcounts of the children in a node cloned via clone_node. Doesn't check if it's a leaf node or
    // not!
    #[inline]
    fn fix_node_refcount(node: usize, words: &mut [u64]) {
        let count = Flags::from_ref(&words[node]).count();
        for i in (node + 1)..=(node + count) {
            let flags = Flags::from_ref_mut(&mut words[words[i] as usize]);
            flags.set_refcount(flags.refcount() + 1);
            flags.set_parity();
        }
    }

    #[inline]
    fn append(offset: usize, words: &mut [u64], index: u8, value: u64, count: usize) {
        words[offset] |= 0b1 << index;
        words[offset + count + 1] = value;

        let node = Flags::from_ref_mut(&mut words[offset]);
        node.set_parity();
        let indice = node.offset(index);

        // Do up to N swaps (going backwards) to get the value in the right cell
        for i in (indice..count).rev() {
            words.swap(offset + i + 1, offset + i + 2);
        }
    }

    #[inline]
    fn remove(offset: usize, words: &mut [u64], index: u8, count: usize) -> u64 {
        let indice = Flags::from_ref_mut(&mut words[offset]).offset(index);
        words[offset] &= !(0b1 << index);
        Flags::from_ref_mut(&mut words[offset]).set_parity();

        // Do up to N swaps (going forwards) to get the value we removed to the end
        for i in indice..(count - 1) {
            words.swap(offset + i + 1, offset + i + 2);
        }

        let value = words[offset + count];
        words[offset + count] = 0;

        value
    }

    pub fn verify_trie(root: u64, bits: usize, words: &[u64]) {
        assert_ne!(root, 0);
        assert_ne!(root, u64::MAX);
        assert_ne!(words[root as usize], 0);
        assert_ne!(words[root as usize], u64::MAX);
        let flags = Flags::from_ref(&words[root as usize]);
        if bits > 5 {
            let count = flags.count();
            for i in 1..=count {
                Self::verify_trie(words[root as usize + i], bits - 5, words);
            }
        }
    }

    fn delete_checked(offset: usize, bits: usize, header: &mut Header, words: &mut [u64], recurse: bool) -> Result<()> {
        assert_ne!(words[offset], 0);
        assert_ne!(words[offset], u64::MAX);
        let node = Flags::from_ref_mut(&mut words[offset]);

        if node.refcount() <= 1 {
            let mut count = node.count();
            assert!(count <= 32); // popcount on a 32-bit integer shouldn't be greater than 32 or math is broken

            if recurse {
                // Decrement and (if necessary) delete any children we have, UNLESS we are a leaf node.
                if bits > 5 {
                    for i in 1..=count {
                        Self::delete_checked(words[offset + i] as usize, bits - 5, header, words, true)?;
                    }
                }
            }

            // Set all words to zero
            words[offset + 1..offset + 1 + count].fill(0);

            // After setting the known cells to zero, walk forward to see if there were any orphaned cells
            while offset + 1 + count < words.len() && words[offset + 1 + count] == 0 {
                count += 1;
            }

            // This should never happen unless something is corrupted
            assert_ne!(count, 0);
            if count == 0 {
                return Err(Error::DirtyTrieState.into());
            }

            // Add it on to the appropriate freelist (0th index has 1 node, so  we use count - 1)
            words[offset] = header.freelist[count - 1];
            header.freelist[count - 1] = offset as u64;
        } else {
            node.set_refcount(node.refcount() - 1);
            node.set_parity();
        }

        Ok(())
    }

    fn insert_checked(
        offset: usize,
        store: &mut RefMut<'_, Storage>,
        count: usize,
        index: u8,
        value: u64,
        mutable: bool,
    ) -> Result<u64> {
        if count == 32 {
            // This can only happen if the key already exists and somehow a previous check failed
            return Err(Error::AlreadyExists(store.words()[offset + index as usize + 1]).into());
        }
        // If we are mutable and we have space, we just append the child and return nothing
        if mutable && offset + count + 1 < store.words().len() && store.words()[offset + count + 1] == 0 {
            Self::append(offset, store.words_mut(), index, value, count);
            Ok(0)
        } else {
            // Otherwise, we must clone ourselves
            let n = store.allocate(count + 1)?;
            let words = store.words_mut();
            Self::clone_node(offset, n, count, words);

            // Then append to the clone, and return the clone
            Self::append(n as usize, words, index, value, count);
            Ok(n)
        }
    }

    fn insert_node(
        offset: usize,
        store: &mut RefMut<'_, Storage>,
        bits: usize,
        key: K,
        value: u64,
        mutable: bool,
    ) -> Result<u64> {
        let node = Flags::from_ref_mut(&mut store.words_mut()[offset]);
        let mutable = mutable && node.refcount() == 1;
        let count = node.count();
        if bits > 5 {
            let index: u8 = key.shr(bits - 5).as_() & 0b11111;

            // Check if the child exists already
            if node.exists(index) {
                // If it does exist, just recurse into it
                let child_offset = offset + node.offset(index) + 1;
                let child = store.words()[child_offset];
                let n = Self::insert_node(child as usize, store, bits - 5, key, value, mutable)?;

                // If it returns a new node, that means it replaced itself.
                if n != 0 {
                    if mutable {
                        let (header, words) = store.parts_mut();
                        // In this case, we can delete the old node without touching the refcounts of the new one
                        Self::delete_checked(child as usize, bits - 5, header, words, false)?;

                        // Then we point to the new node offset
                        words[child_offset] = n;

                        Ok(0)
                    } else {
                        // Otherwise, we have to properly increment the refcounts of the cloned node, then clone
                        // ourselves
                        let clone = store.allocate(count)?;
                        let words = store.words_mut();
                        if bits - 5 > 5 {
                            Self::fix_node_refcount(n as usize, words);
                        }

                        Self::clone_node(offset, clone, count, words);
                        let clone_node = Flags::from_ref(&words[clone as usize]);

                        // Then we point our clone to the new node offset and return it
                        words[clone as usize + clone_node.offset(index) + 1] = n;
                        Ok(clone)
                    }
                } else {
                    Ok(0)
                }
            } else {
                // Create a new child node that is empty and recurse into it. We force it to be mutable, so it shouldn't
                // return anything.
                let child = store.allocate(1)?;
                store.words_mut()[child as usize] = Flags::new().with_refcount(1).with_leaf(bits <= 10).into();
                let n = Self::insert_node(child as usize, store, bits - 5, key, value, true)?;
                assert_eq!(n, 0);

                Self::insert_checked(offset, store, count, index, child, mutable)
            }
        } else {
            let index: u8 = key.as_() & (0b11111 >> (5 - bits));
            // Check if the child exists already
            if node.exists(index) {
                let child_offset = node.offset(index);
                let words = store.words();
                Err(Error::AlreadyExists(words[offset + child_offset + 1]).into())
            } else {
                Self::insert_checked(offset, store, count, index, value, mutable)
            }
        }
    }

    pub fn insert(&mut self, key: K, value: u64) -> Result<Option<HashedArrayTrie<K>>> {
        let mut store = self.storage.borrow_mut();

        let offset: usize = self.offset as usize;
        let bits = size_of::<K>() * 8;
        let mutable = {
            let (_, words) = store.parts();
            let root = Flags::from_ref(&words[offset]);
            root.refcount() <= 1
        };

        let n = Self::insert_node(offset, &mut store, bits, key, value, mutable)?;
        if n != 0 {
            // We reacquire root here after the call so that Rust doesn't think we require two borrows at the same time
            let (_, words) = store.parts_mut();
            let root = Flags::from_ref(&words[offset]);
            if root.refcount() > 1 {
                Self::fix_node_refcount(n as usize, words);
                return Ok(Some(HashedArrayTrie {
                    storage: Rc::clone(&self.storage),
                    offset: n,
                    phantomkey: PhantomData,
                }));
            } else {
                let index: u8 = key.shr(bits - 5).as_() & 0b11111;
                Self::append(offset, words, index, n, root.count());
            }
        }

        Ok(None)
    }

    pub fn get(&self, key: K) -> Result<u64> {
        let store = self.storage.borrow();
        let mut offset: usize = self.offset as usize;
        let mut bits = size_of::<K>() * 8;
        let words = store.words();

        while bits > 5 {
            let node = Flags::from_ref(&words[offset]);
            let index: u8 = key.shr(bits - 5).as_() & 0b11111;

            // Check if the node is set
            if !node.exists(index) {
                return Err(Error::NotFound {
                    key_offset: bits as u8,
                    key_bits: index,
                    node: offset as u64,
                }
                .into());
            }

            offset = words[offset + node.offset(index) + 1] as usize;
            bits -= 5;
        }

        let node = Flags::from_ref(&words[offset]);
        let index: u8 = key.as_() & (0b11111 >> (5 - bits));

        // Check if the node is set
        if !node.exists(index) {
            return Err(Error::NotFound {
                key_offset: bits as u8,
                key_bits: index,
                node: offset as u64,
            }
            .into());
        }

        Ok(words[offset + node.offset(index) + 1])
    }

    fn delete_node(offset: usize, store: &mut RefMut<'_, Storage>, bits: usize, key: K) -> Result<u64> {
        let node = Flags::from_ref_mut(&mut store.words_mut()[offset]);
        if bits > 5 {
            let index: u8 = key.shr(bits - 5).as_() & 0b11111;

            // Check if the node is set
            if !node.exists(index) {
                return Err(Error::NotFound {
                    key_offset: bits as u8,
                    key_bits: index,
                    node: offset as u64,
                }
                .into());
            }

            let child_offset = offset + node.offset(index) + 1;
            let count = node.count();
            let child = store.words()[child_offset] as usize;
            let value = Self::delete_node(child, store, bits - 5, key)?;

            if Flags::from_ref(&store.words()[child]).count() == 0 {
                let (header, words) = store.parts_mut();
                let check = Self::remove(offset, words, index, count);
                assert_eq!(check, child as u64);
                Self::delete_checked(child, bits - 5, header, words, true)?;
            }

            Ok(value)
        } else {
            let index: u8 = key.as_() & (0b11111 >> (5 - bits));

            // Check if the node is set
            if !node.exists(index) {
                return Err(Error::NotFound {
                    key_offset: bits as u8,
                    key_bits: index,
                    node: offset as u64,
                }
                .into());
            }

            let count = node.count();
            let (_, words) = store.parts_mut();
            let value = Self::remove(offset, words, index, count);
            Ok(value)
        }
    }

    pub fn delete(&mut self, key: K) -> Result<u64> {
        let mut store = self.storage.borrow_mut();

        let offset: usize = self.offset as usize;
        let bits = size_of::<K>() * 8;
        Self::delete_node(offset, &mut store, bits, key)
    }
}

#[cfg(not(miri))]
impl Drop for Storage {
    fn drop(&mut self) {
        // Set our clean close bit to 1 and flush.
        let (header, _) = self.parts_mut();
        header.clean = 1;
        // We have to ignore any errors here because we're already in the process of closing everything.
        let _ = self.flush();
    }
}

#[cfg(miri)]
impl Drop for Storage {
    fn drop(&mut self) {
        let (header, _) = self.parts_mut();
        header.clean = 1;

        if let Some(mapref) = &mut self.mapping {
            unsafe {
                let mapping = std::mem::take(mapref);
                let maplen = mapping.len();
                let mapcap = mapping.capacity();
                let _ = Vec::<u64>::from_raw_parts(
                    mapping.leak().as_mut_ptr() as *mut u64,
                    maplen / size_of::<u64>(),
                    mapcap / size_of::<u64>(),
                );
            }
        }
    }
}

#[cfg(test)]
use rand::seq::SliceRandom;
#[cfg(test)]
use rand::thread_rng;
#[cfg(not(miri))]
#[cfg(test)]
use tempfile::tempfile;

#[cfg(test)]
#[inline]
fn get_next_u128<T>(rng: &mut T) -> u128
where
    T: rand::RngCore,
{
    (rng.next_u64() as u128) | rng.next_u64() as u128
}

#[cfg(not(miri))]
#[test]
fn test_storage_new() -> Result<()> {
    let _ = Storage::new_file(tempfile()?, 64)?;
    Ok(())
}

#[cfg(miri)]
#[test]
fn test_storage_new() -> Result<()> {
    let _ = Storage::new(Path::new(""), 64)?;
    Ok(())
}

#[test]
fn test_out_of_storage() -> Result<()> {
    #[cfg(not(miri))]
    let fileref = tempfile()?;
    #[cfg(not(miri))]
    let mut store = Storage::new_ref(&fileref, 32)?;
    #[cfg(miri)]
    let mut store = Storage::new(Path::new(""), 32)?;

    store.allocate(1)?;
    let e = store.allocate(32).expect_err("Should have run out of memory?!");
    assert_eq!(e.downcast::<Error>().unwrap(), Error::OutOfMemory(576));
    Ok(())
}

#[test]
fn test_storage_allocate() -> Result<()> {
    #[cfg(not(miri))]
    let mut store = Storage::new_file(tempfile()?, 32)?;
    #[cfg(miri)]
    let mut store = Storage::new(Path::new(""), 32)?;

    for i in 1..=32 {
        while let Err(e) = store.allocate(i) {
            match e.downcast::<Error>().unwrap() {
                Error::OutOfMemory(_) => store.resize(),
                err => Err(err.into()),
            }?;
        }
    }
    Ok(())
}

#[test]
#[cfg(not(miri))]
fn test_storage_load() -> Result<()> {
    let fileref = tempfile()?;

    {
        let mut store = Storage::new_ref(&fileref, 64)?;
        store.allocate(1)?;
        store.allocate(1)?;
    }

    {
        let mut store = Storage::load_file(fileref)?;
        for i in 1..=32 {
            while let Err(e) = store.allocate(i) {
                match e.downcast::<Error>().unwrap() {
                    Error::OutOfMemory(_) => store.resize(),
                    err => Err(err.into()),
                }?;
            }
        }
    }

    Ok(())
}

#[test]
fn test_empty() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 32)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 32)?));

    // We will eventually put a 128-bit FullLogID struct in here
    let _: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);
    Ok(())
}

#[test]
fn test_deep_near_miss() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1024)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1024)?));

    let mut trie: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);
    trie.insert(1, 1)?;
    trie.insert(2, 2)?;
    assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    assert_eq!(trie.get(2).expect("Failed to get key"), 2);
    Ok(())
}

#[test]
fn test_duplicate() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1024)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1024)?));

    let mut trie: HashedArrayTrie<u16> = HashedArrayTrie::new(&storage, 1);
    trie.insert(1, 1)?;
    assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    let e = trie.insert(1, 2).expect_err("Should have been an error!");
    assert_eq!(e.downcast::<Error>().unwrap(), Error::AlreadyExists(1));
    assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    Ok(())
}

#[test]
fn test_delete() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1024)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1024)?));

    let mut trie: HashedArrayTrie<u16> = HashedArrayTrie::new(&storage, 1);
    trie.insert(1, 1)?;
    assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    assert_eq!(trie.delete(1).expect("Failed to delete key"), 1);
    let e = trie.get(1).expect_err("Should have been an error!");
    assert_eq!(
        e.downcast::<Error>().unwrap(),
        Error::NotFound {
            key_bits: 0, // This should fail immediately, due to the root being empty
            key_offset: 16,
            node: 1
        }
    );
    Ok(())
}

#[test]
fn test_near_miss() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1024)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1024)?));

    let mut trie: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);
    trie.insert(1, 1)?;
    trie.insert(1 << 126, 2)?;
    assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    assert_eq!(trie.get(1 << 126).expect("Failed to get key"), 2);
    Ok(())
}

#[test]
fn test_fill_leaf() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1 << 13)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1 << 13)?));

    let mut trie: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);
    for i in 0..32 {
        trie.insert(i, i as u64)?;
    }

    for i in 0..32 {
        assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
    }
    Ok(())
}

#[test]
fn test_fill_node() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1 << 13)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1 << 13)?));

    let mut trie: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);

    for i in 0..32 {
        trie.insert(i << 6, i as u64 + 10000)?;
        HashedArrayTrie::<u128>::verify_trie(1, 128, trie.storage.as_ref().borrow().words());
    }

    for i in 0..32 {
        assert_eq!(trie.get(i << 6).expect("Failed to get key"), i as u64 + 10000);
    }

    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_fill_trie() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 1 << 22)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 1 << 16)?));

    #[cfg(not(miri))]
    const MAX_COUNT: u16 = u16::MAX;
    #[cfg(miri)]
    const MAX_COUNT: u8 = u8::MAX;

    // Fill a 16-bit trie with with every single possible key (8-bit for miri tests)
    #[cfg(not(miri))]
    let mut trie: HashedArrayTrie<u16> = HashedArrayTrie::new(&storage, 1);
    #[cfg(miri)]
    let mut trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);
    #[cfg(not(miri))]
    let mut v: Vec<u16> = (0..=MAX_COUNT).collect();
    #[cfg(miri)]
    let mut v: Vec<u8> = (0..=MAX_COUNT).collect();

    //let mut rng = StdRng::seed_from_u64(42);
    let mut rng = thread_rng();
    v.shuffle(&mut rng);
    for i in &v {
        trie.insert(*i, *i as u64).expect("Insertion failure!");
    }

    // Verify they all exist
    for i in 0..=MAX_COUNT {
        assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
    }

    v.shuffle(&mut rng);
    // Remove and re-insert every single possible value
    for i in &v {
        trie.delete(*i)?;
        trie.insert(*i, *i as u64)?;
    }

    // Verify all values are correct
    for i in 0..=MAX_COUNT {
        assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
    }

    v.shuffle(&mut rng);
    // Remove all of them in a random order
    for i in &v {
        assert_eq!(trie.delete(*i)?, *i as u64);
    }

    v.shuffle(&mut rng);
    // Re-insert them in a different order
    for i in &v {
        trie.insert(*i, *i as u64).expect("Insertion failure!");
    }

    // Verify all values are correct
    for i in 0..=MAX_COUNT {
        assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
    }

    // Remove them all again
    for i in 0..=MAX_COUNT {
        assert_eq!(trie.delete(i)?, i as u64);
    }

    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_fill_random() -> Result<()> {
    #[cfg(not(miri))]
    let storage = Rc::new(RefCell::new(Storage::new_file(tempfile()?, 32)?));
    #[cfg(miri)]
    let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), 32)?));

    let mut trie: HashedArrayTrie<u128> = HashedArrayTrie::new(&storage, 1);

    let mut rng = thread_rng();
    let mut track: Vec<u128> = Vec::new();

    #[cfg(not(miri))]
    const MAX_COUNT: usize = 0xFFFF;
    #[cfg(miri)]
    const MAX_COUNT: usize = 64;

    // Fill trie with random noise
    for _ in 0..=MAX_COUNT {
        let key = get_next_u128(&mut rng);
        track.push(key);
        while let Err(e) = trie.insert(key, key as u64) {
            match e.downcast::<Error>().unwrap() {
                Error::OutOfMemory(_) => storage.borrow_mut().resize(),
                err => Err(err.into()),
            }?;
        }
    }

    storage.borrow_mut().flush()?;

    // Remove and re-insert every key
    for i in &track {
        assert_eq!(trie.delete(*i)?, *i as u64);
        trie.insert(*i, *i as u64)?;
    }

    storage.borrow_mut().flush()?;

    // Remove everything first, then re-insert every key, ensuring no additional space is used
    for i in &track {
        assert_eq!(trie.delete(*i)?, *i as u64);
    }

    storage.borrow_mut().flush()?;

    for i in &track {
        trie.insert(*i, *i as u64)?;
    }

    storage.borrow_mut().flush()?;

    // Verify all values are correct
    for i in &track {
        assert_eq!(trie.get(*i).expect("Failed to get key"), *i as u64);
    }

    storage.borrow_mut().flush()?;

    //delete everything one more time
    for i in track {
        assert_eq!(trie.delete(i)?, i as u64);
    }

    storage.borrow_mut().flush()?;

    Ok(())
}

// This is too expensive for miri
#[test]
#[cfg_attr(miri, ignore)]
fn test_allocations() -> Result<()> {
    for sz in 1..=512 {
        #[cfg(not(miri))]
        let storage = Rc::new(RefCell::new(Storage::new_file(
            tempfile()?,
            sz * size_of::<u64>() as u64,
        )?));
        #[cfg(miri)]
        let storage = Rc::new(RefCell::new(Storage::new(Path::new(""), sz * size_of::<u64>() as u64)?));

        let mut trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);

        for i in 0..=255 {
            while let Err(e) = trie.insert(i, i as u64) {
                match e.downcast::<Error>().unwrap() {
                    Error::OutOfMemory(_) => storage.borrow_mut().resize(),
                    err => Err(err.into()),
                }?;
            }
        }
        for i in 0..=255 {
            assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
        }
    }
    Ok(())
}

#[cfg(not(miri))]
#[test]
fn test_storage_restore_simple() -> Result<()> {
    let fileref = tempfile()?;

    {
        let storage = Rc::new(RefCell::new(Storage::new_ref(&fileref, 296)?));
        let mut trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);

        trie.insert(1, 1).expect("Insertion failure!");
        assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    }

    {
        let storage = Rc::new(RefCell::new(Storage::restore_file(fileref)?));
        let trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);
        assert_eq!(trie.get(1).expect("Failed to get key"), 1);
    }

    Ok(())
}

#[cfg(not(miri))]
#[test]
fn test_storage_restore() -> Result<()> {
    let fileref = tempfile()?;

    {
        let storage = Rc::new(RefCell::new(Storage::new_ref(&fileref, 1 << 16)?));
        let mut trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);
        // Fill an 8-bit trie with with every single possible key
        let mut v: Vec<u8> = (0..=u8::MAX).collect();

        //let mut rng = StdRng::seed_from_u64(42);
        let mut rng = thread_rng();
        v.shuffle(&mut rng);
        for i in &v {
            trie.insert(*i, *i as u64).expect("Insertion failure!");
        }

        for i in 0..=u8::MAX {
            assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
        }
    }

    {
        let storage = Rc::new(RefCell::new(Storage::restore_file(fileref)?));
        let trie: HashedArrayTrie<u8> = HashedArrayTrie::new(&storage, 1);

        for i in 0..=u8::MAX {
            assert_eq!(trie.get(i).expect("Failed to get key"), i as u64);
        }
    }

    Ok(())
}
