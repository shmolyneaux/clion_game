use facet::Facet;
use facet_json::from_str;
use facet_testhelpers::test;

#[test]
fn transparent_string() {
    let markup = r#"
        "I look like a string"
    "#;

    let s: String = from_str(markup)?;
    assert_eq!(s, "I look like a string");
}

#[test]
fn transparent_tuple_struct() {
    let markup = r#"
        "I look like a string"
    "#;

    #[derive(Facet, Clone, Debug)]
    #[facet(transparent)]
    struct MyString(String);

    let t: MyString = from_str(markup)?;
    assert_eq!(t.0, "I look like a string".to_string());
}

#[test]
fn transparent_non_zero_u64_with_42_value() {
    use std::num::NonZeroU64;

    let markup = r#"
        42
    "#;

    // Test deserialization of NonZeroU64
    let number: NonZeroU64 = from_str(markup)?;
    assert_eq!(number, NonZeroU64::new(42).unwrap());
}

#[test]
fn transparent_non_zero_u64_with_zero_value() {
    use std::num::NonZeroU64;

    let markup = r#"
        0
    "#;

    // Test deserializing 0 into NonZeroU64, which should fail
    let result: Result<NonZeroU64, _> = from_str(markup);
    assert!(result.is_err());
}

#[test]
fn transparent_arc_string() {
    use std::sync::Arc;

    let markup = r#"
        "I'm in an Arc"
    "#;

    // Test deserializing directly into Arc<String>
    let arc_string: Arc<String> = from_str(markup)?;
    assert_eq!(*arc_string, "I'm in an Arc".to_string());
}

#[test]
fn transparent_option_string() {
    let markup = r#"
        "I'm optional"
    "#;

    // Test deserializing a JSON string into Option<String>
    let opt: Option<String> = from_str(markup)?;
    assert_eq!(opt, Some("I'm optional".to_string()));
}

#[test]
fn transparent_option_non_zero_u64_some() {
    use std::num::NonZeroU64;

    // Test deserializing a valid non-zero value
    let markup = r#"
        10
    "#;
    let opt_num: Option<NonZeroU64> = from_str(markup)?;
    assert_eq!(opt_num, Some(NonZeroU64::new(10).unwrap()));
}

#[test]
fn transparent_option_non_zero_u64_none() {
    use std::num::NonZeroU64;

    // Test deserializing a null into Option<NonZeroU64>, which should yield None
    let markup = r#"
        null
    "#;
    let opt_none: Option<NonZeroU64> = from_str(markup)?;
    assert_eq!(opt_none, None);
}

#[test]
fn transparent_option_non_zero_u16_some() {
    use std::num::NonZeroU16;

    // Test deserializing a valid non-zero value
    let markup = r#"
        10
    "#;
    let opt_num: Option<NonZeroU16> = from_str(markup)?;
    assert_eq!(opt_num, Some(NonZeroU16::new(10).unwrap()));
}

#[test]
fn transparent_option_non_zero_u16_none() {
    use std::num::NonZeroU16;

    // Test deserializing a null into Option<NonZeroU16>, which should yield None
    let markup = r#"
        null
    "#;
    let opt_none: Option<NonZeroU16> = from_str(markup)?;
    assert_eq!(opt_none, None);
}

#[test]
fn transparent_string_as_map_key() {
    use std::collections::HashMap;

    #[derive(Facet, Clone, Debug, PartialEq, Eq, Hash)]
    #[facet(transparent)]
    struct UserId(String);

    let markup = r#"
        {
            "user123": "Alice",
            "user456": "Bob",
            "user789": "Charlie"
        }
    "#;

    let map: HashMap<UserId, String> = from_str(markup)?;
    assert_eq!(
        map.get(&UserId("user123".to_string())),
        Some(&"Alice".to_string())
    );
    assert_eq!(
        map.get(&UserId("user456".to_string())),
        Some(&"Bob".to_string())
    );
    assert_eq!(
        map.get(&UserId("user789".to_string())),
        Some(&"Charlie".to_string())
    );
    assert_eq!(map.len(), 3);
}
