use facet_json::from_str;
use facet_testhelpers::test;

#[test]
fn test_debug_format_for_errors() {
    let result = from_str::<i32>("x");
    let err = result.unwrap_err();

    let debug_str = format!("{:?}", err);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_with_rich_diagnostics() {
    let result = from_str::<i32>("x");
    let err = result.unwrap_err();

    // This should trigger the rich diagnostics display code
    let display_str = format!("{}", err);

    #[cfg(not(miri))]
    insta::assert_snapshot!(display_str);
}
