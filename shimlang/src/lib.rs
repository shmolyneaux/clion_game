#![allow(dead_code)]

#[cfg(feature = "facet")]
use facet::Facet;

pub mod parse;
pub mod lex;
pub mod compile;
#[macro_use]
pub mod mem;
pub mod runtime;
pub mod shimlibs;

pub use parse::*;
pub use lex::*;
pub use compile::*;
pub use mem::*;
pub use runtime::*;
