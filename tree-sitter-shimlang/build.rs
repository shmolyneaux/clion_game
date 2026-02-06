fn main() {
    let src_dir = std::path::Path::new("src");

    let mut c_config = cc::Build::new();
    c_config.include(src_dir);
    c_config
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wno-unused-parameter");

    let parser_path = src_dir.join("parser.c");
    c_config.file(&parser_path);

    let scanner_path = src_dir.join("scanner.c");
    c_config.file(&scanner_path);

    c_config.compile("tree-sitter-shimlang");
}
