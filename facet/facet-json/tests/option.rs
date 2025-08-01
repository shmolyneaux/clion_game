use facet::Facet;
use facet_json::from_str;
use facet_testhelpers::test;

#[test]
fn test_from_json_with_option() {
    #[derive(Facet)]
    struct Options {
        name: Option<String>,
        age: Option<u32>,
        inner: Option<Inner>,
    }

    #[derive(Facet)]
    struct Inner {
        foo: i32,
    }

    let json = r#"{
        "name": "Alice",
        "age": null,
        "inner": {
            "foo": 42
        }
    }"#;

    let test_struct: Options = from_str(json)?;
    assert_eq!(test_struct.name.as_deref(), Some("Alice"));
    assert_eq!(test_struct.age, None);
    assert_eq!(test_struct.inner.as_ref().map(|i| i.foo), Some(42));
}

#[test]
fn test_from_json_with_nested_options() {
    #[derive(Facet)]
    struct Options {
        name: Option<Option<String>>,
        age: Option<Box<u32>>,
        inner: Option<Box<Option<Inner>>>,
    }

    #[derive(Facet)]
    struct Inner {
        foo: i32,
    }

    let json = r#"{
        "name": "Alice",
        "age": 5,
        "inner": {
            "foo": 42
        }
    }"#;

    let test_struct: Options = from_str(json)?;
    assert_eq!(test_struct.name.flatten().as_deref(), Some("Alice"));
    assert_eq!(test_struct.age, Some(Box::new(5)));
    assert_eq!(
        test_struct
            .inner
            .and_then(|inner| inner.map(|inner| inner.foo)),
        Some(42)
    );
}
