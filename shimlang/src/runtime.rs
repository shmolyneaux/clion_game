use std::collections::HashMap;
use std::any::{Any, type_name, type_name_of_val};
use std::mem::size_of;
use std::rc::Rc;

use shm_tracy::*;
use shm_tracy::zone_scoped;

use crate::parse::*;
use crate::lex::{debug_u8s, format_script_err};
use crate::compile::*;
use crate::mem::*;
use crate::shimlibs::*;

// Wrapper structure that chains scopes for the environment.
// Variables are stored in a contiguous block of [len: u8][ident_bytes: [u8; len]][value: ShimValue]
// entries stored inline in a single MMU allocation. Lookups scan raw &[u8] bytes directly —
// no allocations, no hashing, no probing.
//
// Each entry occupies 1 + name_len + 8 bytes. For a typical variable name of ~6 bytes, that's
// 15 bytes per entry. A scope starts with capacity 0 and lazily allocates on first insert.
#[derive(Debug)]
pub(crate) struct EnvScope {
    // Pointer to the contiguous data block in MMU (Word(0) when capacity is 0)
    pub(crate) data: Word,
    // Allocated size of the data block in u64 words
    pub(crate) capacity: u32,
    // Used size of the data block in bytes
    used: u32,
    // Pointer to the parent scope in MMU (0 means no parent)
    pub(crate) parent: u24,
    // Depth of this scope in the chain (root is 1)
    depth: u32,
}

// Default capacity when a scope's data block is first allocated (in u64 words).
// 16 words = 128 bytes, enough for ~8 variables with 6-byte names before needing to grow.
const ENV_SCOPE_DEFAULT_CAPACITY: u32 = 16;

impl EnvScope {
    fn new() -> Self {
        Self {
            data: Word(0.into()),
            capacity: 0,
            used: 0,
            parent: 0.into(),
            depth: 1,
        }
    }

    fn new_with_parent(parent_pos: u24, parent_depth: u32) -> Self {
        Self {
            data: Word(0.into()),
            capacity: 0,
            used: 0,
            parent: parent_pos,
            depth: parent_depth + 1,
        }
    }

    /// Get a byte slice view of the used portion of this scope's data block.
    /// Safety: `self.data` must be a valid MMU word pointing to at least `self.capacity`
    /// words, and `self.used` must be <= `self.capacity * 8`.
    pub(crate) unsafe fn raw_bytes<'a>(&self, mem: &'a MMU) -> &'a [u8] {
        if self.used == 0 {
            return &[];
        }
        let start = usize::from(self.data.0);
        let word_count = (self.used as usize).div_ceil(8);
        let u64_slice = &mem.mem[start..start + word_count];
        let ptr = u64_slice.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.used as usize) }
    }

    /// Get a mutable byte slice view of the full capacity of a scope's data block.
    /// Takes explicit data/capacity to avoid borrow conflicts when the EnvScope
    /// reference is obtained via raw pointer.
    unsafe fn raw_bytes_mut_from(mem: &mut MMU, data: Word, capacity: u32) -> &mut [u8] {
        let start = usize::from(data.0);
        let u64_slice = &mut mem.mem[start..start + capacity as usize];
        let ptr = u64_slice.as_mut_ptr() as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, capacity as usize * 8) }
    }

    /// Scan this scope's data block for `key`, returning the byte offset of
    /// the value (ShimValue) within the block, or None if not found.
    /// Layout per entry: [len: u8][ident_bytes: [u8; len]][value: ShimValue (8 bytes)]
    fn scan_for_key(&self, mem: &MMU, key: &[u8]) -> Option<usize> {
        let bytes = unsafe { self.raw_bytes(mem) };
        scan_for_key(bytes, key)
    }

    /// Write a ShimValue at the given byte offset within this scope's data block.
    /// Safety: `value_offset + 8` must be within capacity.
    unsafe fn write_value_at(mem: &mut MMU, data: Word, capacity: u32, value_offset: usize, val: ShimValue) {
        unsafe {
            let buf = EnvScope::raw_bytes_mut_from(mem, data, capacity);
            let val_bytes: [u8; 8] = std::mem::transmute(val);
            std::ptr::copy_nonoverlapping(val_bytes.as_ptr(), buf[value_offset..].as_mut_ptr(), 8);
        }
    }

    /// Reallocate the data block to `new_capacity` words, copying `used` bytes of
    /// existing data. Frees the old block if `capacity > 0`. Returns the new data pointer.
    fn realloc(mem: &mut MMU, data: Word, capacity: u32, used: u32, new_capacity: u32) -> Word {
        let new_data = alloc!(mem, Word(new_capacity.into()), "EnvScope data grow");
        // Copy old data
        if used > 0 {
            let old_start = usize::from(data.0);
            let new_start = usize::from(new_data.0);
            let old_word_count = (used as usize).div_ceil(8);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mem.mem.as_ptr().add(old_start),
                    mem.mem.as_mut_ptr().add(new_start),
                    old_word_count,
                );
            }
        }
        // Free old block (only if there was one)
        if capacity > 0 {
            mem.free(data, Word(capacity.into()));
        }
        new_data
    }
}

/// Scan a contiguous scope data block (as raw bytes) for `key`, returning the byte
/// offset of the value (ShimValue) within the block, or None if not found.
fn scan_for_key(bytes: &[u8], key: &[u8]) -> Option<usize> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let entry_key_len = bytes[offset] as usize;
        let entry_key_start = offset + 1;
        let entry_key_end = entry_key_start + entry_key_len;
        let value_offset = entry_key_end;
        // Each entry is 1 + key_len + 8 bytes
        let entry_end = value_offset + 8;
        if entry_end > bytes.len() {
            break;
        }
        if entry_key_len == key.len() && &bytes[entry_key_start..entry_key_end] == key {
            return Some(value_offset);
        }
        offset = entry_end;
    }
    None
}

#[derive(Debug)]
pub struct Environment {
    // Points to the current EnvScope in MMU
    // u32 is used as u24 converted to u32, 0 means no scope (empty environment)
    current_scope: u32,
}

impl Environment {
    pub fn new(mem: &mut MMU) -> Self {
        // Allocate an EnvScope wrapper (data block allocated lazily on first insert)
        let scope_pos = mem.alloc_and_set(EnvScope::new(), "EnvScope");

        Self {
            current_scope: scope_pos.0.into(),
        }
    }

    pub fn with_scope(captured_scope: u32) -> Self {
        Self {
            current_scope: captured_scope,
        }
    }

    pub fn new_with_builtins(interpreter: &mut Interpreter) -> Self {
        let mut env = Self::new(&mut interpreter.mem);
        let builtins: &[(&[u8], Box<NativeFn>)] = &[
            (b"print", Box::new(shim_print)),
            (b"panic", Box::new(shim_panic)),
            (b"dict", Box::new(shim_dict)),
            (b"Range", Box::new(shim_range)),
            (b"assert", Box::new(shim_assert)),
            (b"str", Box::new(shim_str)),
            (b"int", Box::new(shim_int)),
            (b"float", Box::new(shim_float)),
            (b"try_int", Box::new(shim_try_int)),
            (b"try_float", Box::new(shim_try_float)),
        ];

        for (name, func) in builtins {
            let position = interpreter.mem.alloc_and_set(**func, &format!("builtin func {}", debug_u8s(name)));
            env.insert_new(interpreter, name.to_vec(), ShimValue::NativeFn(position));
        }

        env
    }

    fn insert_new(&mut self, interpreter: &mut Interpreter, key: Vec<u8>, val: ShimValue) {
        assert!(key.len() <= u8::MAX as usize, "Key length {} exceeds maximum {}", key.len(), u8::MAX);

        // Check if key already exists in the current scope — update in place (upsert)
        let scope: &EnvScope = unsafe { interpreter.mem.get(Word(self.current_scope.into())) };
        if let Some(value_offset) = scope.scan_for_key(&interpreter.mem, &key) {
            let (data, capacity) = (scope.data, scope.capacity);
            unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, value_offset, val); }
            return;
        }

        // Read current scope header via raw pointer to avoid borrow issues
        let (data, capacity, used) = unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            ((*scope_ptr).data, (*scope_ptr).capacity, (*scope_ptr).used)
        };

        // Key not found — append new entry
        let entry_size = 1 + key.len() + 8; // len byte + ident bytes + ShimValue
        let new_used = used as usize + entry_size;

        // Grow if needed (also handles initial allocation when capacity == 0)
        let (data, capacity) = if new_used > capacity as usize * 8 {
            let mut new_capacity = if capacity == 0 { ENV_SCOPE_DEFAULT_CAPACITY } else { capacity * 2 };
            while new_used > new_capacity as usize * 8 {
                new_capacity *= 2;
            }
            let new_data = EnvScope::realloc(&mut interpreter.mem, data, capacity, used, new_capacity);
            (new_data, new_capacity)
        } else {
            (data, capacity)
        };

        // Update scope header (data/capacity may have changed)
        unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            (*scope_ptr).data = data;
            (*scope_ptr).capacity = capacity;
        }

        // Append entry: [len: u8][ident_bytes][value: ShimValue (8 bytes)]
        unsafe {
            let buf = EnvScope::raw_bytes_mut_from(&mut interpreter.mem, data, capacity);
            let off = used as usize;
            buf[off] = key.len() as u8;
            buf[off + 1..off + 1 + key.len()].copy_from_slice(&key);
        }
        unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, used as usize + 1 + key.len(), val); }

        // Update used in scope header
        unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            (*scope_ptr).used = new_used as u32;
        }
    }

    fn update(&mut self, interpreter: &mut Interpreter, key: &[u8], val: ShimValue) -> Result<(), String> {
        // Walk the scope chain to find the key
        let mut current_scope_pos = self.current_scope;

        loop {
            if current_scope_pos == 0 {
                break;
            }

            let (parent, data, capacity, value_offset) = unsafe {
                let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                (scope.parent, scope.data, scope.capacity, scope.scan_for_key(&interpreter.mem, key))
            };

            if let Some(value_offset) = value_offset {
                unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, value_offset, val); }
                return Ok(());
            }

            current_scope_pos = parent.into();
        }

        Err(format!("Key {:?} not found in environment", key))
    }

    fn get(&self, interpreter: &mut Interpreter, key: &[u8]) -> Option<ShimValue> {
        let mut current_scope_pos = self.current_scope;

        loop {
            if current_scope_pos == 0 {
                break;
            }

            let (parent, value_offset) = unsafe {
                let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                (scope.parent, scope.scan_for_key(&interpreter.mem, key))
            };

            if let Some(value_offset) = value_offset {
                // Read the ShimValue from the byte offset
                let val: ShimValue = unsafe {
                    let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                    let bytes = scope.raw_bytes(&interpreter.mem);
                    let mut val_bytes = [0u8; 8];
                    std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                    std::mem::transmute(val_bytes)
                };
                return Some(val);
            }

            current_scope_pos = parent.into();
        }

        None
    }

    fn contains_key(&self, interpreter: &mut Interpreter, key: &[u8]) -> bool {
        self.get(interpreter, key).is_some()
    }

    fn push_scope(&mut self, mem: &mut MMU) {
        // Get current scope depth
        let current_depth = if self.current_scope == 0 {
            0
        } else {
            let current: &EnvScope = unsafe {
                mem.get(Word(self.current_scope.into()))
            };
            current.depth
        };
        
        // Allocate a new EnvScope with parent pointing to current scope
        // (data block allocated lazily on first insert)
        let scope_pos = mem.alloc_and_set(
            EnvScope::new_with_parent(self.current_scope.into(), current_depth),
            "EnvScope"
        );
        
        // Update current scope to the new one
        self.current_scope = scope_pos.0.into();
    }

    fn pop_scope(&mut self, mem: &MMU) -> Result<(), String> {
        if self.current_scope == 0 {
            return Err(format!("Ran out of scopes to pop!"));
        }
        
        // Get the current EnvScope
        let scope: &EnvScope = unsafe {
            mem.get(Word(self.current_scope.into()))
        };
        
        // Move to parent scope
        let parent: u32 = scope.parent.into();
        if parent == 0 {
            return Err(format!("Cannot pop root scope!"));
        }
        
        self.current_scope = parent;
        Ok(())
    }
    
    // Helper to get the depth of the current scope
    fn scope_depth(&self, mem: &MMU) -> usize {
        if self.current_scope == 0 {
            return 0;
        }
        
        let scope: &EnvScope = unsafe {
            mem.get(Word(self.current_scope.into()))
        };
        scope.depth as usize
    }
}

// TODO: If we do NaN-boxing we could have f64 (rather than f32) for "free"
#[derive(Copy, Clone, Debug)]
pub enum ShimValue {
    Uninitialized,
    Unit,
    None,
    Integer(i32),
    Float(f32),
    Bool(bool),
    // Memory position pointing to ShimFn structure
    Fn(Word),
    BoundMethod(
        // Object
        Word,
        // Fn memory position pointing to ShimFn structure
        Word,
    ),
    BoundNativeMethod(
        // ShimValue followed by NativeFn
        Word,
    ),
    // A function pointer doesn't fit in the ShimValue, so we need to store the
    // function pointer in interpreter memory
    NativeFn(Word),
    // TODO: it seems like this should point to a more generic reference-counted
    // object type that all non-value types share
    String(
        // len
        u16,
        // byte offset within the 8-byte aligned word
        u8,
        // position (word index into memory)
        u24,
    ),
    List(Word),
    Dict(Word),
    StructDef(Word),
    Struct(Word),
    Native(Word),
    // For now this is really only used for GC purposes
    Environment(Word),
}
const _: () = {
    assert!(std::mem::size_of::<ShimValue>() == 8);
};

pub(crate) trait ShimNative: Any {
    fn to_string(&self, _interpreter: &mut Interpreter) -> String {
        format!("{}", type_name::<Self>())
    }

    fn get_attr(&self, _self_as_val: &ShimValue, _interpreter: &mut Interpreter, _ident: &[u8]) -> Result<ShimValue, String> {
        Err(format!("Can't get_attr on {}", type_name::<Self>() ))
    }

    fn set_attr(
        &self,
        _interpreter: &mut Interpreter,
        _ident: &[u8],
        _val: ShimValue,
    ) -> Result<(), String> {
        Err(format!("Can't set_attr on {}", type_name::<Self>() ))
    }

    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn gc_vals(&self) -> Vec<ShimValue>;
}

pub(crate) type NativeFn = fn(&mut Interpreter, &ArgBundle) -> Result<ShimValue, String>;
const _: () = {
    assert!(std::mem::size_of::<NativeFn>() == 8);
};

pub(crate) fn format_float(val: f32) -> String {
    let s = format!("{val}");
    if !s.contains('.') && !s.contains('e') {
        format!("{s}.0")
    } else {
        s
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum StructAttribute {
    MemberInstanceOffset(u8),
    MethodDef(Word),
}

#[derive(Debug)]
pub(crate) struct StructDef {
    name: Vec<u8>,
    pub(crate) member_count: u8,
    lookup: Vec<(Vec<u8>, StructAttribute)>,
}

// Stores function information in interpreter memory
pub(crate) struct ShimFn {
    // Program counter where the function code begins
    pub(crate) pc: u32,
    // Length of the function name string
    pub(crate) name_len: u16,
    // Memory position of the function name (stored as string)
    pub(crate) name: Word,
    // The environment scope where this function was defined (for closures)
    pub(crate) captured_scope: u32,
}

const _: () = {
    assert!(std::mem::size_of::<ShimFn>() == 16);
};

impl StructDef {
    fn find(&self, ident: &[u8]) -> Option<StructAttribute> {
        for (attr, loc) in self.lookup.iter() {
            if ident == attr {
                return Some(*loc)
            }
        }
        None
    }

    pub(crate) fn mem_size(&self) -> usize {
        // TODO: if the StructDef changes it might be effectively non const sized
        // in interpreter memory
        const _: () = {
            assert!(std::mem::size_of::<StructDef>() == 56);
        };
        std::mem::size_of::<StructDef>() / 8
    }
}

#[derive(Debug)]
pub(crate) enum CallResult {
    ReturnValue(ShimValue),
    PC(u32, u32), // PC and captured_scope
}

#[derive(Debug)]
pub struct ArgBundle {
    pub(crate) args: Vec<ShimValue>,
    pub(crate) kwargs: Vec<(Ident, ShimValue)>,
}

impl ArgBundle {
    pub fn new() -> Self {
        Self {
            args: Vec::new(),
            kwargs: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.args.len() + self.kwargs.len()
    }

    fn clear(&mut self) {
        self.args.clear();
        self.kwargs.clear();
    }
}

pub(crate) struct ArgUnpacker<'a> {
    bundle: &'a ArgBundle,
    pos: usize,
    kwargs_consumed: usize,
}

impl<'a> ArgUnpacker<'a> {
    pub(crate) fn new(bundle: &'a ArgBundle) -> Self {
        Self { bundle, pos: 0, kwargs_consumed: 0 }
    }

    pub(crate) fn required(&mut self, name: &[u8]) -> Result<ShimValue, String> {
        self.optional(name).ok_or_else(|| format!("Missing required argument: '{}'", debug_u8s(name)))
    }

    pub(crate) fn optional(&mut self, name: &[u8]) -> Option<ShimValue> {
        for (ident, arg) in self.bundle.kwargs.iter() {
            if ident == name {
                self.kwargs_consumed += 1;
                return Some(*arg);
            }
        }
        // Return next positional argument
        match self.bundle.args.get(self.pos) {
            Some(val) => {
                self.pos += 1;
                Some(*val)
            },
            None => None,
        }
    }

    pub(crate) fn end(&self) -> Result<(), String> {
        let consumed = self.pos + self.kwargs_consumed;
        if self.bundle.len() != consumed {
            Err(format!("Got {} arguments, but only used {}", self.bundle.len(), consumed))
        } else {
            Ok(())
        }
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn fnv1a_hash(key: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

macro_rules! numeric_op {
    ($lhs:tt $op:tt $rhs:expr) => {
        match ($lhs, $rhs) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(*a $op *b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(*a $op *b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) $op *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a $op (*b as f32))),
            (a, b) => Err(format!(
                "Operation '{}' not supported between {:?} and {:?}",
                stringify!($op), a, b
            )),
        }
    };
}

impl ShimValue {
    pub(crate) fn is_uninitialized(&self) -> bool {
        if let ShimValue::Uninitialized = self {
            true
        } else {
            false
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, ShimValue::None)
    }

    pub(crate) fn hash(&self, interpreter: &mut Interpreter) -> Result<u32, String> {
        let hashcode: u64 = match self {
            ShimValue::Integer(i) => fnv1a_hash(&i.to_be_bytes()),
            ShimValue::Float(f) => fnv1a_hash(&f.to_be_bytes()),
            ShimValue::String(..) => {
                fnv1a_hash(&self.string(interpreter).unwrap().to_vec())
            },
            // We might want to salt these to reduce collisions with other type,
            // but I expect there is a fairly trivial difference in performance
            // and would imply heterogenous dicts.
            ShimValue::None => fnv1a_hash(&[0x00]),
            ShimValue::Bool(false) => fnv1a_hash(&[0x00]),
            ShimValue::Bool(true) => fnv1a_hash(&[0x01]),
            _ => return Err(format!("Can't hash {:?}", self))
        };

        Ok(hashcode as u32)
    }

    pub(crate) fn as_native<T: ShimNative>(&self, interpreter: &mut Interpreter) -> Result<&mut T, String> {
        match self {
            ShimValue::Native(position) => unsafe {
                let boxobj: &mut Box<dyn ShimNative> =
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);

                let mutboxobj = boxobj.as_any_mut();
                let name = type_name_of_val(mutboxobj);
                match mutboxobj.downcast_mut::<T>() {
                    Some(obj) => Ok(obj),
                    _ => Err(format!("Can't get {} as {}", name, type_name::<T>()))
                }
            },
            _ => Err(format!("Can't try_into non-native {:?}", self))
        }
    }

    pub(crate) fn call(
        &self,
        interpreter: &mut Interpreter,
        args: &mut ArgBundle,
    ) -> Result<CallResult, String> {
        match self {
            ShimValue::None => Err(format!("Can't call None as a function")),
            ShimValue::Fn(fn_pos) => {
                let shim_fn: &ShimFn = unsafe { interpreter.mem.get(*fn_pos) };
                Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope))
            }
            ShimValue::BoundMethod(pos, fn_pos) => {
                // push struct pos to start of arg list then return the pc of the method
                args.args.insert(0, ShimValue::Struct(*pos));
                let shim_fn: &ShimFn = unsafe { interpreter.mem.get(*fn_pos) };
                Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope))
            }
            ShimValue::BoundNativeMethod(pos) => {
                let obj: &ShimValue = unsafe { interpreter.mem.get(*pos) };
                let native_fn: &NativeFn = unsafe { interpreter.mem.get(*pos + 1) };

                args.args.insert(0, *obj);
                Ok(CallResult::ReturnValue(native_fn(interpreter, args)?))
            }
            ShimValue::StructDef(struct_def_pos) => {
                let struct_def: &StructDef = unsafe { interpreter.mem.get(*struct_def_pos) };
                if struct_def.member_count as usize != args.len() || !args.kwargs.is_empty()  {
                    // Call the internal __init__ to handle default/kw arguments
                    // If we're not using defaults we could handle kw arguments here,
                    // but for now it simplifies things to push all the special cases to __init__
                    if let Some(StructAttribute::MethodDef(fn_pos)) = struct_def.find(b"__init__") {
                        let shim_fn: &ShimFn = unsafe { interpreter.mem.get(fn_pos) };
                        return Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope));
                    } else {
                        return Err(format!("INTERNAL: no __init__ on StructDef"));
                    }
                }

                // Allocate space for each member, plus the header
                let word_count = Word((struct_def.member_count as u32 + 1).into());
                let new_pos = alloc!(
                    interpreter.mem,
                    word_count,
                    "Struct instantiation"
                );

                // The first word points to the StructDef
                interpreter.mem.mem[usize::from(new_pos.0)] = u64::from(struct_def_pos.0);

                // The remaining words get copies of the arguments to the initializer
                for (idx, arg) in args.args.iter().enumerate() {
                    interpreter.mem.mem[usize::from(new_pos.0) + 1 + idx] = arg.to_u64();
                }

                Ok(CallResult::ReturnValue(ShimValue::Struct(new_pos)))
            }
            ShimValue::NativeFn(pos) => {
                let native_fn: &NativeFn = unsafe { interpreter.mem.get(*pos) };
                Ok(CallResult::ReturnValue(native_fn(interpreter, args)?))
            }
            other => Err(format!(
                "Can't call value {:?} as a function",
                other.to_string(interpreter)
            )),
        }
    }

    pub(crate) fn dict_mut(&self, interpreter: &mut Interpreter) -> Result<&mut ShimDict, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &mut ShimDict = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };
                Ok(dict)
            },
            _ => {
                Err(format!("Not a dict"))
            }
        }
    }

    pub(crate) fn dict(&self, interpreter: &Interpreter) -> Result<&ShimDict, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &ShimDict = unsafe {
                    std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)])
                };
                Ok(dict)
            },
            _ => {
                Err(format!("Not a dict"))
            }
        }
    }

    pub(crate) fn list_mut(&self, interpreter: &mut Interpreter) -> Result<&mut ShimList, String> {
        match self {
            ShimValue::List(position) => {
                unsafe {
                    Ok(std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]))
                }
            },
            _ => {
                Err(format!("Not a list"))
            }
        }
    }

    pub(crate) fn list(&self, interpreter: &Interpreter) -> Result<&ShimList, String> {
        match self {
            ShimValue::List(position) => {
                unsafe {
                    Ok(std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)]))
                }
            },
            _ => {
                Err(format!("Not a list"))
            }
        }
    }

    fn native(&self, interpreter: &mut Interpreter) -> Result<&mut Box<dyn ShimNative>, String> {
        match self {
            ShimValue::Native(position) => unsafe {
                let ptr: *mut Box<dyn ShimNative> =
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                Ok(&mut *ptr)
            },
            _ => {
                Err(format!("Not a native"))
            }
        }
    }

    fn expect_string(&self, interpreter: &Interpreter) -> &[u8] {
        self.string(interpreter).unwrap()
    }

    pub(crate) fn string(&self, interpreter: &Interpreter) -> Result<&[u8], String> {
        match self {
            ShimValue::String(len, offset, position) => {
                let len = *len as usize;
                let offset = *offset as usize;
                let position_usize = usize::from(*position);
                let total_len: usize = (offset + len).div_ceil(8);

                let bytes: &[u8] = unsafe {
                    let u64_slice = &interpreter.mem.mem[
                        position_usize..
                        (position_usize+total_len)
                    ];
                    std::slice::from_raw_parts(
                        (u64_slice.as_ptr() as *const u8).add(offset),
                        len,
                    )
                };
                Ok(bytes)
            },
            _ => {
                Err(format!("Not a string"))
            }
        }
    }

    pub(crate) fn integer(&self) -> Result<i32, String> {
        match self {
            ShimValue::Integer(i) => Ok(*i),
            _ => Err(format!("Not an integer")),
        }
    }

    fn index(&self, interpreter: &mut Interpreter, index: &ShimValue) -> Result<ShimValue, String> {
        match (self, index) {
            (ShimValue::String(..), ShimValue::Integer(index)) => {
                let index = *index as isize;

                let val = self.string(interpreter)?;

                let len = val.len() as isize;
                let index: isize = if index < -len || index >= len {
                    return Err(format!("Index {} is out of bounds", index));
                } else if index < 0 {
                    len + index as isize
                } else {
                    index as isize
                };

                let b: u8 = val[index as usize];

                Ok(interpreter.mem.alloc_str(&[b]))
            },
            (ShimValue::List(position), ShimValue::Integer(idx)) => {
                unsafe {
                    let lst: &ShimList =
                        std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)]);
                    lst.get(&interpreter.mem, *idx as isize)
                }
            },
            (ShimValue::Dict(_), some_key) => {
                let dict = self.dict_mut(interpreter)?;

                dict.get(interpreter, *some_key)
            }
            (a, b) => Err(format!("Can't index {:?} with {:?}", a, b)),
        }
    }

    fn set_index(
        &self,
        interpreter: &mut Interpreter,
        index: &ShimValue,
        value: &ShimValue,
    ) -> Result<(), String> {
        match (self, index) {
            (ShimValue::List(position), ShimValue::Integer(index)) => {
                let index = *index as usize;
                let list: &mut ShimList = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };
                list.set(&mut interpreter.mem, index as isize, *value)?;
                Ok(())
            }
            (ShimValue::Dict(position), index) => {
                let dict: &mut ShimDict = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };

                dict.set(interpreter, *index, *value)
            }
            (a, b) => Err(format!("Can't set index {:?} with {:?}", a, b)),
        }
    }

    fn to_shimvalue_string(&self, interpreter: &mut Interpreter) -> ShimValue {
        let s = self.to_string(interpreter);
        interpreter.mem.alloc_str(&s.into_bytes())
    }

    pub fn to_string(&self, interpreter: &mut Interpreter) -> String {
        match self {
            ShimValue::Uninitialized => format!("Uninitialized"),
            ShimValue::None => "None".to_string(),
            ShimValue::Integer(i) => i.to_string(),
            ShimValue::Float(f) => format_float(*f),
            ShimValue::Bool(false) => "false".to_string(),
            ShimValue::Bool(true) => "true".to_string(),
            ShimValue::String(..) => {
                String::from_utf8(self.string(interpreter).unwrap().to_vec()).expect("valid utf-8 string stored")
            },
            ShimValue::List(_) => {
                let lst = self.list(interpreter).unwrap();

                let mut out = "[".to_string();
                for idx in 0..lst.len() {
                    if idx != 0 {
                        out.push_str(",");
                        out.push_str(" ");
                    }
                    let item = lst.get(&interpreter.mem, idx as isize).unwrap();
                    out.push_str(&item.to_string(interpreter));
                }
                out.push_str("]");

                out
            },
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().to_string(interpreter)
            }
            ShimValue::Struct(pos) => {
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    
                    // Get the struct name
                    let struct_name = debug_u8s(&def.name).to_string();
                    
                    // Collect member names and values first to avoid borrowing issues
                    let mut members: Vec<(String, ShimValue)> = Vec::new();
                    for (attr, loc) in def.lookup.iter() {
                        // Only collect member variables, not methods
                        if let StructAttribute::MemberInstanceOffset(offset) = loc {
                            let attr_name = debug_u8s(attr).to_string();
                            let val: ShimValue = *interpreter.mem.get(*pos + *offset as u32 + 1);
                            members.push((attr_name, val));
                        }
                    }
                    
                    // Build output like "Point(x=2.0, y=3.0)"
                    let mut out = struct_name;
                    out.push('(');
                    
                    for (idx, (attr_name, val)) in members.iter().enumerate() {
                        if idx != 0 {
                            out.push_str(", ");
                        }
                        out.push_str(attr_name);
                        out.push('=');
                        out.push_str(&val.to_string(interpreter));
                    }
                    
                    out.push(')');
                    out
                }
            }
            value => format!("{:?}", value),
        }
    }

    pub(crate) fn is_truthy(&self, interpreter: &mut Interpreter) -> Result<bool, String> {
        match self {
            ShimValue::None => Ok(false),
            ShimValue::Integer(i) => Ok(*i != 0),
            ShimValue::Float(f) => Ok(*f != 0.0),
            ShimValue::Bool(false) => Ok(false),
            ShimValue::Bool(true) => Ok(true),
            ShimValue::String(..) => {
                Ok(!self.expect_string(interpreter).is_empty())
            },
            ShimValue::List(_) => {
                Ok(!self.list(interpreter)?.is_empty())
            },
            _ => Ok(true),
        }
    }

    pub(crate) fn add(&self, interpreter: &mut Interpreter, other: &Self, pending_args: &mut ArgBundle) -> Result<CallResult, String> {
        match (self, other) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(CallResult::ReturnValue(ShimValue::Integer(*a + *b))),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(CallResult::ReturnValue(ShimValue::Float(*a + *b))),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(CallResult::ReturnValue(ShimValue::Float((*a as f32) + *b))),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(CallResult::ReturnValue(ShimValue::Float(*a + (*b as f32)))),
            (a @ ShimValue::String(..), b @ ShimValue::String(..)) => {
                let a = a.string(interpreter)?;
                let b = b.string(interpreter)?;

                let c = interpreter.mem.alloc_str(
                    &format!("{}{}",
                        unsafe { std::str::from_utf8_unchecked(a) },
                        unsafe { std::str::from_utf8_unchecked(b) },
                    ).into_bytes()
                );

                Ok(CallResult::ReturnValue(c))
            }
            (ShimValue::Struct(_), b) => {
                // TODO: why do we need to take in `pending_args` when we could
                // construct a new ArgBundle?
                pending_args.args.clear();
                pending_args.args.push(*b);
                self.get_attr(interpreter, b"add")?.call(interpreter, pending_args)
            },
            (a, b) => Err(format!(
                "Operation '+' not supported between {:?} and {:?}",
                a, b
            )),
        }
    }

    fn sub(&self, other: &Self) -> Result<ShimValue, String> {
        numeric_op!(self - other)
    }

    pub(crate) fn equal_inner(&self, interpreter: &mut Interpreter, other: &Self) -> Result<bool, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(a == b),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(a == b),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(a == b),
            (a @ ShimValue::String(..), b @ ShimValue::String(..)) => {
                let a = a.string(interpreter)?;
                let b = b.string(interpreter)?;
                Ok(a == b)
            }
            (ShimValue::None, ShimValue::None) => Ok(true),
            (a @ ShimValue::List(_), b @ ShimValue::List(_)) => {
                let a = a.list(interpreter)?;
                let b = b.list(interpreter)?;
                if a.len() != b.len() {
                    return Ok(false)
                }
                for idx in 0..a.len() {
                    let item_a = a.get(&interpreter.mem, idx as isize)?;
                    let item_b = b.get(&interpreter.mem, idx as isize)?;
                    if !item_a.equal_inner(interpreter, &item_b)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            _ => Ok(false),
        }
    }

    fn equal(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        Ok(ShimValue::Bool(self.equal_inner(interpreter, other)?))
    }

    fn not_equal(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            _ => Ok(ShimValue::Bool(true)),
        }
    }

    fn mul(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        numeric_op!(self * other)
    }

    fn div(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        // NOTE: All division is floating point division
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a / b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a / b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) / *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a / (*b as f32))),
            (a, b) => Err(format!("Can't Divide {:?} and {:?}", a, b)),
        }
    }

    fn modulus(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        // NOTE: All division is floating point division
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a % b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a % b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) % *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a % (*b as f32))),
            (a, b) => Err(format!("Can't Divide {:?} and {:?}", a, b)),
        }
    }

    pub(crate) fn gt(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => {
                Ok(ShimValue::Bool(*a == true && *b == false))
            }
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) > *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a > (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? > other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Greater) => Ok(ShimValue::Bool(true)),
                    Ok(_) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't GT {:?} and {:?}", a, b)),
        }
    }

    fn gte(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) >= *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a >= (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? >= other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Greater) | Ok(std::cmp::Ordering::Equal) => Ok(ShimValue::Bool(true)),
                    Ok(std::cmp::Ordering::Less) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't GTE {:?} and {:?}", a, b)),
        }
    }

    pub(crate) fn lt(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => {
                Ok(ShimValue::Bool(*a == false && *b == true))
            }
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) < *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a < (*b as f32))),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? < other.string(interpreter)?))
            },
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Less) => Ok(ShimValue::Bool(true)),
                    Ok(_) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't LT {:?} and {:?}", a, b)),
        }
    }

    fn lte(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) <= *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a <= (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? <= other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Less) | Ok(std::cmp::Ordering::Equal) => Ok(ShimValue::Bool(true)),
                    Ok(std::cmp::Ordering::Greater) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't LTE {:?} and {:?}", a, b)),
        }
    }

    fn contains(
        &self,
        interpreter: &mut Interpreter,
        some_key: &Self,
    ) -> Result<ShimValue, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &mut ShimDict = unsafe {
                    let ptr: &mut ShimDict =
                        std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    ptr
                };

                if let Ok(_) = dict.get(interpreter, *some_key) {
                    return Ok(ShimValue::Bool(true));
                } else {
                    return Ok(ShimValue::Bool(false));
                }
            }
            _ => Err(format!("Can't `in` {:?} and {:?}", self, some_key)),
        }
    }

    fn not(&self, interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Bool(a) => Ok(ShimValue::Bool(!a)),
            ShimValue::Float(a) => Ok(ShimValue::Bool(*a == 0.0)),
            ShimValue::Integer(a) => Ok(ShimValue::Bool(*a == 0)),
            ShimValue::None => Ok(ShimValue::Bool(true)),
            ShimValue::List(_) => Ok(ShimValue::Bool(!self.is_truthy(interpreter)?)),
            _ => Ok(ShimValue::Bool(false)),
        }
    }

    fn neg(&self, _interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Float(a) => Ok(ShimValue::Float(-a)),
            ShimValue::Integer(a) => Ok(ShimValue::Integer(-a)),
            _ => Err(format!("Can't Negate {:?}", self)),
        }
    }

    pub(crate) fn get_attr(&self, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        match self {
            ShimValue::Struct(pos) => {
                // Handle __type__ special attribute
                if ident == b"__type__" {
                    unsafe {
                        let def_pos: u64 = *interpreter.mem.get(*pos);
                        let def_pos: Word = Word((def_pos as u32).into());
                        return Ok(ShimValue::StructDef(def_pos));
                    }
                }
                
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    Ok(*interpreter.mem.get(*pos + *offset as u32 + 1))
                                }
                                StructAttribute::MethodDef(fn_pos) => {
                                    // Return the bound method with the pre-allocated function
                                    Ok(ShimValue::BoundMethod(*pos, *fn_pos))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", debug_u8s(ident), self))
            }
            ShimValue::StructDef(def_pos) => {
                // Handle __name__ special attribute
                if ident == b"__name__" {
                    unsafe {
                        let def: &StructDef = interpreter.mem.get(*def_pos);
                        let name = def.name.clone();
                        return Ok(interpreter.mem.alloc_str(&name));
                    }
                }
                
                unsafe {
                    let def: &StructDef = interpreter.mem.get(*def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(_) => Err(format!(
                                    "Can't access member {:?} on StructDef {:?}",
                                    ident, self
                                )),
                                StructAttribute::MethodDef(fn_pos) => {
                                    // Return the pre-allocated method function
                                    Ok(ShimValue::Fn(*fn_pos))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", debug_u8s(ident), self))
            }
            ShimValue::String(..) => {
                let func = match ident {
                    b"len" => shim_str_len,
                    _ => return Err(format!("No ident {:?} on str", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::List(_) => {
                let func = match ident {
                    b"map" => shim_list_map,
                    b"filter" => shim_list_filter,
                    b"len" => shim_list_len,
                    b"iter" => shim_list_iter,
                    b"sort" => shim_list_sort,
                    b"append" => shim_list_append,
                    b"clear" => shim_list_clear,
                    b"extend" => shim_list_extend,
                    b"index" => shim_list_index,
                    b"insert" => shim_list_insert,
                    b"pop" => shim_list_pop,
                    b"sorted" => shim_list_sorted,
                    b"reverse" => shim_list_reverse,
                    b"reversed" => shim_list_reversed,
                    _ => return Err(format!("No ident {:?} on list", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::Dict(_) => {
                let func = match ident {
                    b"set" => shim_dict_index_set,
                    b"get" => shim_dict_index_get,
                    b"has" => shim_dict_index_has,
                    b"len" => shim_dict_len,
                    b"pop" => shim_dict_pop,
                    b"iter" => shim_dict_iter,
                    b"keys" => shim_dict_keys,
                    b"values" => shim_dict_values,
                    b"items" => shim_dict_items,
                    b"shrink_to_fit" => shim_dict_shrink_to_fit,
                    _ => return Err(format!("No ident {:?} on dict", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().get_attr(self, interpreter, ident)
            }
            val => Err(format!("Ident {:?} not available on {:?}", debug_u8s(ident), val)),
        }
    }

    fn set_attr(
        &self,
        interpreter: &mut Interpreter,
        ident: &[u8],
        val: ShimValue,
    ) -> Result<(), String> {
        match self {
            ShimValue::Struct(pos) => {
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    let slot: &mut ShimValue =
                                        interpreter.mem.get_mut(*pos + *offset as u32 + 1);
                                    *slot = val;
                                    Ok(())
                                }
                                StructAttribute::MethodDef(_) => Err(format!(
                                    "Can't assign to struct method {:?} for {:?}",
                                    ident, self
                                )),
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", ident, self))
            }
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().set_attr(interpreter, ident, val)
            }
            val => Err(format!("Ident {:?} not available on {:?}", ident, val)),
        }
    }

    pub(crate) fn to_u64(&self) -> u64 {
        unsafe {
            let mut tmp: u64 = 0;
            // Copy raw bytes of e into tmp
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                &mut tmp as *mut u64 as *mut u8,
                size_of::<Self>(),
            );
            tmp
        }
    }

    pub(crate) fn to_bytes(&self) -> [u8; 8] {
        unsafe { std::mem::transmute(*self) }
    }

    pub(crate) fn from_bytes(bytes: [u8; 8]) -> Self {
        unsafe { std::mem::transmute(bytes) }
    }

    pub(crate) unsafe fn from_u64(data: u64) -> Self {
        unsafe {
            let mut tmp: Self = std::mem::zeroed(); // Will be overwritten
            std::ptr::copy_nonoverlapping(
                &data as *const u64 as *const u8,
                &mut tmp as *mut Self as *mut u8,
                size_of::<Self>(),
            );
            tmp
        }
    }
}

// TODO: uncomment #[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
    pub program: Rc<Program>,
}

impl Interpreter {
    pub fn print_mem(&self) {
        let _zone = zone_scoped!("print_mem");
        let mut count = 0;
        let mut idx = 0;
        for block in self.mem.free_list.iter() {
            while idx < block.pos.into() {
                println!("{:06}: {:016x}", idx, self.mem.mem[idx]);
                idx += 1;
                count += 1
            }

            if count > 100 {
                break;
            }
        }
    }

    pub fn print_env(&self, env: &Environment) {
        let _zone = zone_scoped!("print_env");
        let mut current_scope_pos = env.current_scope;
        let mut idx = 0;
        
        loop {
            if current_scope_pos == 0 {
                break;
            }
            
            println!("Scope {idx}");
            
            // Get the EnvScope
            let scope: &EnvScope = unsafe {
                self.mem.get(Word(current_scope_pos.into()))
            };
            
            // Walk the contiguous data block and print entries
            let bytes = unsafe { scope.raw_bytes(&self.mem) };
            let mut off = 0usize;
            while off < bytes.len() {
                let key_len = bytes[off] as usize;
                let key_bytes = &bytes[off + 1..off + 1 + key_len];
                let value_offset = off + 1 + key_len;
                let val: ShimValue = unsafe {
                    let mut val_bytes = [0u8; 8];
                    std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                    std::mem::transmute(val_bytes)
                };
                println!("{:>12}: {:?}", debug_u8s(key_bytes), val);
                match val {
                    ShimValue::Struct(pos) => {
                        unsafe {
                            let def_pos: u64 = *self.mem.get(pos);
                            let def_pos: Word = Word((def_pos as u32).into());
                            let def: &StructDef = self.mem.get(def_pos);
                            for (attr, loc) in def.lookup.iter() {
                                match loc {
                                    StructAttribute::MemberInstanceOffset(offset) => {
                                        let val: ShimValue = *self.mem.get(pos + *offset as u32 + 1);
                                        println!("                - {} = {:?}", debug_u8s(&attr), val);
                                    },
                                    StructAttribute::MethodDef(_) => (),
                                };
                            }
                        }
                    },
                    ShimValue::StructDef(pos) => {
                        unsafe {
                            let def: &StructDef = self.mem.get(pos);
                            for (attr, loc) in def.lookup.iter() {
                                match loc {
                                    StructAttribute::MemberInstanceOffset(_) => {
                                        println!("                - {}", debug_u8s(&attr));
                                    },
                                    StructAttribute::MethodDef(_) => {
                                        println!("                - {}()", debug_u8s(&attr));
                                    }
                                };
                            }
                        }
                    },
                    _ => (),
                }
                off = value_offset + 8;
            }
            
            // Move to parent scope
            let parent: u32 = scope.parent.into();
            current_scope_pos = parent;
            idx += 1;
        }
    }

    pub fn gc(&mut self, env: &Environment) {
        let _zone = zone_scoped!("GC");
        //self.print_mem();
        //self.print_env(env);
        
        unsafe {
            let _scope: &EnvScope = self.mem.get(Word(env.current_scope.into()));
        }

        let mut roots: Vec<ShimValue> = Vec::new();
        roots.push(ShimValue::Environment(Word(env.current_scope.into())));
        
        // Now create GC and process roots
        let mut gc = {
            let _zone = zone_scoped!("Init GC");
            GC::new(&mut self.mem)
        };
        gc.mark(roots);
        gc.sweep();
    }

    pub fn create(config: &Config, program: Program) -> Self {
        let mmu = MMU::with_capacity(Word((config.memory_space_bytes / 8).into()));

        Self {
            mem: mmu,
            source: HashMap::new(),
            program: Rc::new(program),
        }
    }

    pub fn append_program(&mut self, program: Program) -> Result<(), String> {
        let span_offset = self.program.script.len() as u32;
        Rc::<Program>::get_mut(&mut self.program).unwrap().bytecode.extend(program.bytecode);
        Rc::<Program>::get_mut(&mut self.program).unwrap().spans.extend(
            program.spans.into_iter().map(|span| Span {
                start: span.start + span_offset,
                end: span.end + span_offset,
            })
        );
        Rc::<Program>::get_mut(&mut self.program).unwrap().script.extend(program.script);

        Ok(())
    }

    pub fn execute_bytecode_extended(
        &mut self,
        mod_pc: &mut usize,
        mut pending_args: ArgBundle,
        env: &mut Environment,
    ) -> Result<ShimValue, String> {
        let _zone = zone_scoped!("Execute Bytecode");
        let mut pc = *mod_pc;
        // These are values that are operated on. Expressions push and pop to
        // this stack, return values go on this stack etc.
        let mut stack: Vec<ShimValue> = Vec::new();


        // This is the (PC, loop_info, scope_count, caller_scope, fn_optional_param_names,
        // fn_optional_param_name_idx) call stack
        let mut stack_frame: Vec<(
            // PC
            usize,
            // loop_info
            Vec<(usize, usize, usize)>,
            // scope_count
            usize,
            // caller_scope
            u32,
            // fn_optional_param_names
            Vec<Ident>,
            // fn_optional_param_name_idx
            usize,
        )> = Vec::new();

        // This is the PC of the (start, end, scope_count) of the current loop for the
        // current function
        let mut loop_info: Vec<(usize, usize, usize)> = Vec::new();

        let mut fn_optional_param_name_idx = 0;
        let mut fn_optional_param_names: Vec<Ident> = Vec::new();

        let bytes = &self.program.clone().bytecode;
        while pc < bytes.len() {
            //let _zone = zone_scoped!("Execute Single Instruction");
            match bytes[pc] {
                val if val == ByteCode::Pop as u8 => {
                    stack.pop();
                }
                val if val == ByteCode::Add as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");

                    match a.add(self, &b, &mut pending_args).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })? {
                        CallResult::ReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc, captured_scope) => {
                            stack_frame.push((
                                pc + 1,
                                loop_info.clone(),
                                env.scope_depth(&self.mem),
                                env.current_scope,
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            // Restore the captured environment and push a new scope for function locals
                            env.current_scope = captured_scope;
                            env.push_scope(&mut self.mem);
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                }
                val if val == ByteCode::Sub as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.sub(&b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::Equal as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.equal(self, &b)?);
                }
                val if val == ByteCode::NotEqual as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::NotEqual");
                    let a = stack.pop().expect("Operand for ByteCode::NotEqual");
                    stack.push(a.not_equal(self, &b)?);
                }
                val if val == ByteCode::Multiply as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Multiply");
                    let a = stack.pop().expect("Operand for ByteCode::Multiply");
                    stack.push(a.mul(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::Divide as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Divide");
                    let a = stack.pop().expect("Operand for ByteCode::Divide");
                    stack.push(a.div(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::Modulus as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Modulus");
                    let a = stack.pop().expect("Operand for ByteCode::Modulus");
                    stack.push(a.modulus(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::GT as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::GT");
                    let a = stack.pop().expect("Operand for ByteCode::GT");
                    stack.push(a.gt(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::GTE as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::GTE");
                    let a = stack.pop().expect("Operand for ByteCode::GTE");
                    stack.push(a.gte(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::LT as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::LT");
                    let a = stack.pop().expect("Operand for ByteCode::LT");
                    stack.push(a.lt(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::LTE as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::LTE");
                    let a = stack.pop().expect("Operand for ByteCode::LTE");
                    stack.push(a.lte(self, &b).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?);
                }
                val if val == ByteCode::In as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::In");
                    let a = stack.pop().expect("Operand for ByteCode::In");
                    stack.push(a.contains(self, &b)?);
                }
                val if val == ByteCode::Range as u8 => {
                    let end = stack.pop().expect("Operand for ByteCode::Range");
                    let start = stack.pop().expect("Operand for ByteCode::Range");
                    
                    let range = RangeNative {
                        start: start,
                        end: end,
                    };
                    stack.push(self.mem.alloc_native(range));
                }
                val if val == ByteCode::Not as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Not");
                    stack.push(a.not(self)?);
                }
                val if val == ByteCode::Negate as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Negate");
                    stack.push(a.neg(self)?);
                }
                val if val == ByteCode::Stringify as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Stringify");
                    stack.push(a.to_shimvalue_string(self));
                }
                val if val == ByteCode::LiteralNone as u8 => {
                    stack.push(ShimValue::None);
                }
                val if val == ByteCode::Copy as u8 => {
                    stack.push(*stack.last().expect("non-empty stack"));
                }
                val if val == ByteCode::LoopStart as u8 => {
                    let loop_end = pc + (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                    loop_info.push((pc + 3, loop_end, env.scope_depth(&self.mem)));
                    pc += 2;
                }
                val if val == ByteCode::LoopEnd as u8 => {
                    loop_info.pop().expect("loop end should have loop info");
                }
                val if val == ByteCode::Break as u8 => {
                    let (_, end_pc, scope_count) =
                        loop_info.last().expect("break should have loop info");
                    while env.scope_depth(&self.mem) > *scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    pc = *end_pc;
                    continue;
                }
                val if val == ByteCode::Continue as u8 => {
                    let (start_pc, _, scope_count) =
                        loop_info.last().expect("continue should have loop info");
                    while env.scope_depth(&self.mem) > *scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    pc = *start_pc;
                    continue;
                }
                val if val == ByteCode::UnpackArgs as u8 => {
                    let required_arg_count = bytes[pc + 1] as usize;
                    let optional_arg_count = bytes[pc + 2] as usize;

                    let mut pos_arg_idx = 0;

                    fn_optional_param_names.clear();
                    fn_optional_param_name_idx = 0;

                    // Assign each parameter in the function to something
                    let mut idx = pc + 3;
                    for param_idx in 0..(required_arg_count + optional_arg_count) {
                        let len = bytes[idx];
                        let param_name = &bytes[idx + 1..idx + 1 + len as usize];

                        if param_idx >= required_arg_count {
                            fn_optional_param_names.push(param_name.to_vec());
                        }

                        // If the parameter was provided as a kwarg, set that now
                        let mut set_arg = false;
                        let mut found_idx = None;
                        for (idx, (ident, _val)) in pending_args.kwargs.iter().enumerate() {
                            if ident == param_name {
                                found_idx = Some(idx);
                                break;
                            }
                        }
                        if let Some(idx) = found_idx {
                            let (_ident, val) = pending_args.kwargs.remove(idx);
                            env.insert_new(self, param_name.to_vec(), val);
                            set_arg = true;
                        }

                        // If it wasn't set as a kwarg, assign it the next positional arg
                        if !set_arg {
                            let val = if pos_arg_idx < pending_args.args.len() {
                                pos_arg_idx += 1;
                                pending_args.args[pos_arg_idx - 1]
                            } else {
                                // We ran out of positional args

                                // If we haven't finished assigning the required
                                // arguments then the function wasn't provided
                                // enough and we need to exit
                                if param_idx < required_arg_count {
                                    return Err(format_script_err(
                                        self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                                        &self.program.script,
                                        &format!("Not enough positional args, arg_count: {}, kwarg_count: {}", pending_args.args.len(), pending_args.kwargs.len()),
                                    ));
                                }

                                ShimValue::Uninitialized
                            };
                            env.insert_new(self, param_name.to_vec(), val);
                        }

                        idx += 1 + len as usize;
                    }
                    if pos_arg_idx != pending_args.args.len() {
                        let remaining = pending_args.args.len() - pos_arg_idx;
                        return Err(format_script_err(
                            self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                            &self.program.script,
                            &format!("Too many positional args, {} remaining", remaining),
                        ));
                    }
                    if !pending_args.kwargs.is_empty() {
                        let mut msg = "Unused kwargs remaining:".to_string();
                        for (ident, _) in pending_args.kwargs.iter() {
                            msg.push(' ');
                            msg.push_str(debug_u8s(ident));
                        }
                        return Err(format_script_err(
                            self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                            &self.program.script,
                            &msg,
                        ));
                    }
                    pc = idx;
                    continue;
                }
                val if val == ByteCode::JmpInitArg as u8 => {
                    let optional_param_name = &fn_optional_param_names[fn_optional_param_name_idx];
                    fn_optional_param_name_idx += 1;

                    match env.get(self, optional_param_name) {
                        Some(ShimValue::Uninitialized) => (),
                        Some(_) => {
                            let new_pc =
                                pc + (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                            pc = new_pc;
                            continue;
                        }
                        None => {
                            return Err(format!(
                                "Expected UnpackArgs to set indent that doesn't exist!"
                            ));
                        }
                    }
                    pc += 2;
                }
                val if val == ByteCode::AssignArg as u8 => {
                    let arg_num = bytes[pc + 1] as usize;
                    let optional_param_name = &fn_optional_param_names[arg_num];
                    env.update(self, optional_param_name, stack.pop().unwrap())?;
                    pc += 1;
                }
                val if val == ByteCode::LiteralShimValue as u8 => {
                    let bytes = [
                        bytes[pc + 1],
                        bytes[pc + 2],
                        bytes[pc + 3],
                        bytes[pc + 4],
                        bytes[pc + 5],
                        bytes[pc + 6],
                        bytes[pc + 7],
                        bytes[pc + 8],
                    ];
                    stack.push(ShimValue::from_bytes(bytes));
                    pc += 8;
                }
                val if val == ByteCode::LiteralString as u8 => {
                    let str_len = bytes[pc + 1] as usize;
                    let contents = &bytes[pc + 2..pc + 2 + str_len as usize];

                    stack.push(self.mem.alloc_str(contents));
                    pc += 1 + str_len;
                }
                val if val == ByteCode::VariableDeclaration as u8 => {
                    let val = stack.pop().expect("Value for declaration");
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];
                    env.insert_new(self, ident.to_vec(), val);
                    pc += 1 + ident_len;
                }
                val if val == ByteCode::Assignment as u8 => {
                    let val = stack.pop().expect("Value for assignment");
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    if !env.contains_key(self, ident) {
                        return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &format!("Identifier {:?} not found", ident),
                        ));
                    }
                    env.update(self, ident, val)?;

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::VariableLoad as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];
                    if let Some(value) = env.get(self, ident) {
                        stack.push(value);
                    } else {
                        return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &format!("Unknown identifier {:?}", debug_u8s(ident)),
                        ));
                    }
                    pc += 1 + ident_len;
                }
                val if val == ByteCode::GetAttr as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    let obj = stack.pop().expect("val to access");

                    let res = match obj.get_attr(self, ident) {
                        Ok(val) => val,
                        Err(msg) => return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &msg,
                        )),
                    };


                    stack.push(res);

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::SetAttr as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    let val = stack.pop().expect("val to assign");
                    let obj = stack.pop().expect("obj to set");
                    obj.set_attr(self, ident, val).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::Index as u8 => {
                    let index = stack.pop().expect("index val");
                    let obj = stack.pop().expect("index obj");

                    let val = obj.index(self, &index).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;

                    stack.push(val);
                }
                val if val == ByteCode::SetIndex as u8 => {
                    let val = stack.pop().expect("index assigned val");
                    let index = stack.pop().expect("index index");
                    let obj = stack.pop().expect("index obj");

                    obj.set_index(self, &index, &val).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;
                }
                val if val == ByteCode::Call as u8 => {
                    let arg_count = bytes[pc + 1];
                    let kwarg_count = bytes[pc + 2];

                    pending_args.clear();

                    for _ in 0..kwarg_count {
                        let val = stack.pop().unwrap();
                        let ident = match stack.pop().unwrap() {
                            val @ ShimValue::String(..) => {
                                val.string(self)?.to_vec()
                            },
                            other => return Err(format!("Invalid kwarg ident {:?}", other)),
                        };
                        pending_args.kwargs.push((ident, val));
                    }

                    for _ in 0..arg_count {
                        pending_args.args.push(stack.pop().unwrap());
                    }
                    pending_args.args.reverse();
                    pending_args.kwargs.reverse();

                    let callable = stack.pop().expect("callable not on stack");

                    match callable.call(self, &mut pending_args).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })? {
                        CallResult::ReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc, captured_scope) => {
                            stack_frame.push((
                                pc + 3,
                                loop_info.clone(),
                                env.scope_depth(&self.mem),
                                env.current_scope,
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            // Restore the captured environment and push a new scope for function locals
                            env.current_scope = captured_scope;
                            env.push_scope(&mut self.mem);
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                    pc += 2;
                }
                val if val == ByteCode::StartScope as u8 => {
                    env.push_scope(&mut self.mem);
                }
                val if val == ByteCode::EndScope as u8 => {
                    env.pop_scope(&self.mem)?;
                }
                val if val == ByteCode::Return as u8 => {
                    if stack_frame.is_empty() {
                        // We're assuming that we were called to run just a
                        // particular function

                        // There should be a single value on that stack that we return
                        if stack.len() != 1 {
                            return Err(format!("Expected one element on stack: {stack:?}"));
                        }

                        // TODO: we should supply `pc` as a `&mut usize`, but
                        // that requires changing far too much code here that
                        // works with `pc` as a value.
                        *mod_pc = pc;
                        return Ok(stack[0]);
                    }

                    // The value at the top of the stack is the return value of
                    // the function, so we just need to pop the PC
                    let scope_count;
                    let caller_scope;
                    (
                        pc,
                        loop_info,
                        scope_count,
                        caller_scope,
                        fn_optional_param_names,
                        fn_optional_param_name_idx,
                    ) = stack_frame.pop().expect("stack frame to return to");
                    while env.scope_depth(&self.mem) > scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    // Restore the caller's environment scope
                    env.current_scope = caller_scope;
                    continue;
                }
                val if val == ByteCode::JmpUp as u8 => {
                    let new_pc = pc - (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::Jmp as u8 => {
                    // TODO: signed jumps
                    let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::JmpNZ as u8 => {
                    let conditional = stack.pop().expect("JMPNZ val to check");
                    if conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::JmpZ as u8 => {
                    let conditional = stack.pop().expect("JMP val to check");
                    if !conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::CreateList as u8 => {
                    let len = ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;

                    let lst_val = self.mem.alloc_list();
                    let lst = lst_val.list_mut(self)?;
                    for item in stack.drain(stack.len() - len..) {
                        lst.push(&mut self.mem, item);
                    }

                    stack.push(lst_val);

                    pc += 2;
                }
                val if val == ByteCode::CreateFn as u8 => {
                    let instruction_offset = ((bytes[pc + 1] as u32) << 8) + bytes[pc + 2] as u32;
                    let fn_pc = pc as u32 - instruction_offset;
                    // Use descriptive name for anonymous functions
                    // Capture the current environment scope
                    let fn_val = self.mem.alloc_fn(fn_pc, b"<anonymous>", env.current_scope);
                    stack.push(fn_val);
                    pc += 2;
                }
                val if val == ByteCode::CreateStruct as u8 => {
                    // Everything after the first two bytes is data for the
                    // struct definition.
                    let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;

                    let member_count = bytes[pc + 3];
                    let method_count = bytes[pc + 4];

                    let mut idx = pc + 5;
                    
                    // Read struct name
                    let name_len = bytes[idx];
                    let name = bytes[idx + 1..idx + 1 + name_len as usize].to_vec();
                    idx = idx + 1 + name_len as usize;

                    let mut struct_table = Vec::new();

                    for member_idx in 0..member_count {
                        let ident_len = bytes[idx];
                        let ident = &bytes[idx + 1..idx + 1 + ident_len as usize];
                        struct_table.push((
                            ident.to_vec(),
                            StructAttribute::MemberInstanceOffset(member_idx),
                        ));
                        idx = idx + 1 + ident_len as usize;
                    }

                    for _ in 0..method_count {
                        let method_pc = pc + ((bytes[idx] as usize) << 8) + bytes[idx + 1] as usize;

                        idx += 2;

                        let ident_len = bytes[idx];
                        let ident = &bytes[idx + 1..idx + 1 + ident_len as usize];
                        
                        // Allocate a function object for this method
                        // Methods capture the environment where the struct is defined
                        let fn_val = self.mem.alloc_fn(method_pc as u32, ident, env.current_scope);
                        let fn_pos = match fn_val {
                            ShimValue::Fn(pos) => pos,
                            _ => panic!("alloc_fn should return Fn"),
                        };
                        
                        struct_table.push((
                            ident.to_vec(),
                            StructAttribute::MethodDef(fn_pos),
                        ));
                        idx = idx + 1 + ident_len as usize;
                    }
                    const _: () = {
                        assert!(std::mem::size_of::<StructDef>() == 56);
                    };
                    let pos = alloc!(
                        self.mem,
                        Word(7.into()),
                        &format!("ByteCode::CreateStruct def PC {pc}")
                    );

                    unsafe {
                        let ptr: *mut StructDef =
                            std::mem::transmute(&mut self.mem.mem[usize::from(pos.0)]);
                        ptr.write(StructDef {
                            name,
                            member_count,
                            lookup: struct_table,
                        });
                    }

                    // Then push the struct definition to the stack
                    stack.push(ShimValue::StructDef(pos));

                    pc = new_pc;
                    continue;
                }
                b => {
                    print_asm(bytes);
                    return Err(format!("Unknown bytecode {b} at PC {pc}"));
                }
            }
            pc += 1;
        }

        *mod_pc = pc;
        if stack.len() > 0 {
            Ok(stack.pop().unwrap())
        } else {
            Ok(ShimValue::Uninitialized)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24_conversion() {
        assert_eq!(u24::from(1u32), u24([0, 0, 1]));
        assert_eq!(u32::from(u24::from(1u32)), 1u32);

        assert_eq!(u24::from(1u32).0, [0, 0, 1]);
    }

    #[test]
    fn scan_for_key_empty() {
        let bytes: &[u8] = &[];
        assert_eq!(scan_for_key(bytes, b"x"), None);
    }

    #[test]
    fn scan_for_key_single_entry() {
        // Entry: [3] "foo" [8 bytes value]
        let mut data = vec![3u8]; // len
        data.extend_from_slice(b"foo");
        data.extend_from_slice(&[0xAA; 8]); // value placeholder
        assert!(scan_for_key(&data, b"foo").is_some());
        assert_eq!(scan_for_key(&data, b"foo"), Some(4)); // offset of value
        assert_eq!(scan_for_key(&data, b"bar"), None);
    }

    #[test]
    fn scan_for_key_multiple_entries() {
        let mut data = Vec::new();
        // Entry 1: "ab" -> 8 bytes
        data.push(2u8);
        data.extend_from_slice(b"ab");
        data.extend_from_slice(&[0x11; 8]);
        // Entry 2: "cde" -> 8 bytes
        data.push(3u8);
        data.extend_from_slice(b"cde");
        data.extend_from_slice(&[0x22; 8]);

        assert_eq!(scan_for_key(&data, b"ab"), Some(3));
        // entry1 = 1+2+8 = 11 bytes, entry2: len at 11, key at 12..15, value at 15
        assert_eq!(scan_for_key(&data, b"cde"), Some(15));
        assert_eq!(scan_for_key(&data, b"xyz"), None);
    }

    fn test_interpreter() -> Interpreter {
        let config = Config::default();
        let program = Program {
            bytecode: Vec::new(),
            spans: Vec::new(),
            script: Vec::new(),
        };
        Interpreter::create(&config, program)
    }

    #[test]
    fn env_scope_insert_and_get() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"x".to_vec(), ShimValue::Integer(42));
        let val = env.get(&mut interpreter, b"x");
        assert!(val.is_some());
        match val.unwrap() {
            ShimValue::Integer(42) => {},
            other => panic!("Expected Integer(42), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_update() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"y".to_vec(), ShimValue::Integer(1));
        env.update(&mut interpreter, b"y", ShimValue::Integer(99)).unwrap();
        match env.get(&mut interpreter, b"y").unwrap() {
            ShimValue::Integer(99) => {},
            other => panic!("Expected Integer(99), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_parent_lookup() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"root_var".to_vec(), ShimValue::Integer(10));
        env.push_scope(&mut interpreter.mem);
        env.insert_new(&mut interpreter, b"child_var".to_vec(), ShimValue::Integer(20));

        // Can see child var
        match env.get(&mut interpreter, b"child_var").unwrap() {
            ShimValue::Integer(20) => {},
            other => panic!("Expected Integer(20), got {:?}", other),
        }
        // Can see parent var through scope chain
        match env.get(&mut interpreter, b"root_var").unwrap() {
            ShimValue::Integer(10) => {},
            other => panic!("Expected Integer(10), got {:?}", other),
        }

        // Pop scope and child var is gone
        env.pop_scope(&interpreter.mem).unwrap();
        assert!(env.get(&mut interpreter, b"child_var").is_none());
        match env.get(&mut interpreter, b"root_var").unwrap() {
            ShimValue::Integer(10) => {},
            other => panic!("Expected Integer(10), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_grow() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        // Insert enough variables to force at least one grow
        for i in 0..20u8 {
            let name = format!("var_{}", i);
            env.insert_new(&mut interpreter, name.into_bytes(), ShimValue::Integer(i as i32));
        }
        // Verify all are retrievable
        for i in 0..20u8 {
            let name = format!("var_{}", i);
            match env.get(&mut interpreter, name.as_bytes()).unwrap() {
                ShimValue::Integer(v) if v == i as i32 => {},
                other => panic!("Expected Integer({}), got {:?}", i, other),
            }
        }
    }
}

/**
 *
 * Struct Bytecode Format
 *  - CreateStruct OpCode
 *    - Two byte relative jump to end of struct def
 *    - u8 member count
 *    - u8 method count
 *    - List of members
 *      - u8 len followed by that number of bytes for the ident
 *    - List of methods
 *      - u16 relative jump to method, u8 len, ident bytes
 *    - Method defs
 *
 * Struct Instance Data Format:
 *  - Header value that points to object metadata
 *    - Contains mapping of ident to member offset or method PC
 *  - Member 0
 *  - Member 1
 *  - ...
 *
 * Struct Metadata Format:
 *  - Just a list for now
 *    - Vec<(Vec<u8>, Offset | PC)>
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */
const _TODO: u8 = 42;

