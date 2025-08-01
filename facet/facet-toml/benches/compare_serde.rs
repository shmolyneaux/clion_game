#![allow(clippy::approx_constant)]

use divan::{Bencher, black_box};
use facet::Facet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested0 {
    id: u64,
    name: String,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested1 {
    id: u64,
    name: String,
    child: Nested0,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested2 {
    id: u64,
    name: String,
    child: Nested1,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested3 {
    id: u64,
    name: String,
    child: Nested2,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested4 {
    id: u64,
    name: String,
    child: Nested3,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested5 {
    id: u64,
    name: String,
    child: Nested4,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested6 {
    id: u64,
    name: String,
    child: Nested5,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested7 {
    id: u64,
    name: String,
    child: Nested6,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested8 {
    id: u64,
    name: String,
    child: Nested7,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested9 {
    id: u64,
    name: String,
    child: Nested8,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested10 {
    id: u64,
    name: String,
    child: Nested9,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested11 {
    id: u64,
    name: String,
    child: Nested10,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested12 {
    id: u64,
    name: String,
    child: Nested11,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested13 {
    id: u64,
    name: String,
    child: Nested12,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested14 {
    id: u64,
    name: String,
    child: Nested13,
}
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
pub struct Nested15 {
    id: u64,
    name: String,
    child: Nested14,
}

// Wide Structure
#[derive(Debug, PartialEq, Clone, Facet, Serialize, Deserialize)]
struct Wide {
    field01: String,
    field02: u64,
    field03: i32,
    field04: f64,
    field05: bool,
    field06: Option<String>,
    field07: Vec<u32>,
    field08: String,
    field09: u64,
    field10: i32,
    field11: f64,
    field12: bool,
    field13: Option<String>,
    field14: Vec<u32>,
    field15: String,
    field16: u64,
    field17: i32,
    field18: f64,
    field19: bool,
    field20: Option<String>,
    field21: Vec<u32>,
    field22: String,
    field23: u64,
    field24: i32,
    field25: f64,
    field26: bool,
    field27: Option<String>,
    field28: Vec<u32>,
    field29: HashMap<String, i32>,
    field30: Nested0,
}

fn create_wide() -> Wide {
    let mut map = HashMap::new();
    map.insert("a".to_string(), 1);
    map.insert("b".to_string(), 2);

    Wide {
        field01: "value 01".to_string(),
        field02: 12345678901234567,
        field03: -123456789,
        field04: 3.141592653589793,
        field05: true,
        field06: Some("optional value 06".to_string()),
        field07: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        field08: "value 08".to_string(),
        field09: 98765432109876543,
        field10: 987654321,
        field11: 2.718281828459045,
        field12: false,
        field13: None,
        field14: vec![0, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        field15: "value 15".to_string(),
        field16: 1111111111111111111,
        field17: -111111111,
        field18: 1.618033988749895,
        field19: true,
        field20: Some("optional value 20".to_string()),
        field21: vec![10, 20, 30],
        field22: "value 22".to_string(),
        field23: 2222222222222222222,
        field24: -222222222,
        field25: 0.5772156649015329,
        field26: false,
        field27: None,
        field28: vec![],
        field29: map,
        field30: Nested0 {
            id: 0,
            name: "Base Nested".to_string(),
        },
    }
}

// Wide benchmark functions

#[divan::bench(name = "Wide - facet_serialize")]
fn bench_wide_facet_serialize(bencher: Bencher) {
    let data = create_wide();
    toml::to_string(&data).expect("Failed to create wide TOML");

    bencher.bench(|| black_box(facet_toml::to_string(black_box(&data))).unwrap());
}

#[divan::bench(name = "Wide - serde_serialize")]
fn bench_wide_serde_serialize(bencher: Bencher) {
    let data = create_wide();
    toml::to_string(&data).expect("Failed to create wide TOML");

    bencher.bench(|| black_box(toml::to_string(black_box(&data))).unwrap());
}

#[divan::bench(name = "Wide - facet_deserialize")]
fn bench_wide_facet_deserialize(bencher: Bencher) {
    let data = create_wide();
    let toml_string = toml::to_string(&data).expect("Failed to create wide TOML");

    bencher.bench(|| {
        let res: Wide = black_box(facet_toml::from_str(black_box(&toml_string))).unwrap();
        black_box(res)
    });
}

#[divan::bench(name = "Wide - serde_deserialize")]
fn bench_wide_serde_deserialize(bencher: Bencher) {
    let data = create_wide();
    let toml_string = toml::to_string(&data).expect("Failed to create wide TOML");

    bencher.bench(|| {
        let res: Wide = black_box(toml::from_str(black_box(&toml_string))).unwrap();
        black_box(res)
    });
}

fn main() {
    divan::main();
}
