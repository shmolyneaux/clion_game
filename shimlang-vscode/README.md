# Shimlang VS Code Extension

Syntax highlighting for the [Shimlang](../shimlang/) programming language (`.shm` files) in Visual Studio Code.

## Features

- Line comments (`//`) and nested block comments (`/* */`)
- String literals with escape sequences (`\n`, `\t`, `\\`, `\"`, `\'`)
- Nested string interpolation — code inside `\(expr)` blocks is highlighted as code, not as a string
- Keywords, constants (`true`, `false`, `None`), and operators
- Numeric literals (integers and floats)
- Function and struct definitions
- `self` keyword

## Usage

1. Open VS Code and go to **Extensions** → **…** → **Install from VSIX…**, or
2. Copy (or symlink) this `shimlang-vscode` folder into your VS Code extensions directory:
   - **Linux/macOS:** `~/.vscode/extensions/shimlang-vscode`
   - **Windows:** `%USERPROFILE%\.vscode\extensions\shimlang-vscode`
3. Reload VS Code. Any `.shm` file will automatically use Shimlang syntax highlighting.

## Running Tests

The grammar tests use [`vscode-tmgrammar-test`](https://github.com/nickmain/vscode-tmgrammar-test). To run them:

```sh
cd tests
npm install
npm test
```

Test cases live in `tests/cases/` as `.shm` files with inline assertion comments that verify TextMate scopes.

## File Structure

```
shimlang-vscode/
├── package.json                  # VS Code extension manifest
├── language-configuration.json   # Bracket matching, comment toggling
├── syntaxes/
│   └── shimlang.tmLanguage.json  # TextMate grammar
└── tests/
    ├── package.json              # Test runner config
    └── cases/                    # Grammar test files
        ├── comments.shm
        ├── strings.shm
        ├── keywords.shm
        ├── numbers_constants.shm
        ├── functions_structs.shm
        ├── operators.shm
        └── nested_features.shm
```
