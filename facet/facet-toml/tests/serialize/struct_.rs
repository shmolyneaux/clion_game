//! Tests for TOML table values.

use facet::Facet;
use facet_testhelpers::test;

use crate::assert_serialize;

#[test]
fn test_table_to_struct() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        value: i32,
        table: Table,
    }

    #[derive(Debug, Facet, PartialEq)]
    struct Table {
        value: i32,
    }

    assert_serialize!(
        Root,
        Root {
            value: 1,
            table: Table { value: 2 },
        },
    );
}

#[test]
fn test_unit_struct() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        value: i32,
        unit: Unit,
    }

    #[derive(Debug, Facet, PartialEq)]
    struct Unit(i32);

    assert_serialize!(
        Root,
        Root {
            value: 1,
            unit: Unit(2),
        },
    );
}

#[test]
fn test_nested_unit_struct() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        value: i32,
        unit: NestedUnit,
    }

    #[derive(Debug, Facet, PartialEq)]
    struct NestedUnit(Unit);

    #[derive(Debug, Facet, PartialEq)]
    struct Unit(i32);

    assert_serialize!(
        Root,
        Root {
            value: 1,
            unit: NestedUnit(Unit(2)),
        },
    );
}

#[test]
fn test_root_struct_multiple_fields() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        a: i32,
        b: Option<bool>,
        c: String,
    }

    assert_serialize!(
        Root,
        Root {
            a: 1,
            b: Some(true),
            c: "'' \"test ".to_string()
        },
    );
}

#[test]
fn test_nested_struct_multiple_fields() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        nested: Nested,
    }

    #[derive(Debug, Facet, PartialEq)]
    struct Nested {
        a: i32,
        b: bool,
        c: String,
    }

    assert_serialize!(
        Root,
        Root {
            nested: Nested {
                a: 1,
                b: true,
                c: "test".to_string()
            }
        },
    );
}

#[test]
fn test_rename_single_struct_fields() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        #[facet(rename = "1")]
        a: i32,
        #[facet(rename = "with spaces")]
        b: bool,
        #[facet(rename = "'quoted'")]
        c: String,
        #[facet(rename = "")]
        d: usize,
    }

    assert_serialize!(
        Root,
        Root {
            a: 1,
            b: true,
            c: "quoted".parse()?,
            d: 2
        },
    );
}

#[test]
fn test_rename_all_struct_fields() {
    #[derive(Debug, Facet, PartialEq)]
    #[facet(rename_all = "kebab-case")]
    struct Root {
        a_number: i32,
        another_bool: bool,
        #[facet(rename = "Overwrite")]
        shouldnt_matter: f32,
    }

    assert_serialize!(
        Root,
        Root {
            a_number: 1,
            another_bool: true,
            shouldnt_matter: 1.0
        },
    );
}

#[test]
fn test_default_struct_fields() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        #[facet(default)]
        a: i32,
        #[facet(default)]
        b: bool,
        #[facet(default)]
        c: String,
    }

    assert_serialize!(
        Root,
        Root {
            a: i32::default(),
            b: bool::default(),
            c: "hi".to_string()
        },
    );
}

#[test]
fn test_optional_default_struct_fields() {
    #[derive(Debug, Facet, PartialEq)]
    struct Root {
        #[facet(default)]
        a: Option<i32>,
        #[facet(default)]
        b: Option<bool>,
        #[facet(default = Some("hi".to_owned()))]
        c: Option<String>,
    }

    assert_serialize!(
        Root,
        Root {
            a: None,
            b: Some(bool::default()),
            c: Some("hi".to_string())
        },
    );
}
