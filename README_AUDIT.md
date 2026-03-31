# README Coverage Audit — `shmolyneaux`

_Generated: 2026-03-31_

This report audits the README coverage for every public repository owned by
`shmolyneaux`. Each repo is checked for four quality sections:

| Column | Meaning |
|---|---|
| **README?** | A README file exists and is non-empty |
| **Desc** | Project description / overview / about section |
| **Build** | Build / install / getting-started / compilation instructions |
| **Tests** | Test / testing instructions |
| **History** | Changelog / release notes / development history |

`✅` = present   `❌` = absent   `⚠️` = file exists but essentially empty

---

## Summary Table

| Repository | README? | Desc | Build | Tests | History | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| [basic-cpp-project-template] | ✅ | ✅ | ✅ | ❌ | ❌ | Missing test and changelog sections |
| [bean_counter] | ✅ | ✅ | ✅ | ❌ | ❌ | Build docs cover dependency patches only; no run or test instructions; no changelog |
| [bevy_simon] | ✅ | ❌ | ✅ | ❌ | ❌ | README is only a `# Dev` section with a single cargo command; no description, tests, or history |
| [clion_game] | ✅ | ✅ | ✅ | ❌ | ❌ | Missing tests and history |
| [hanzi-study-list-generator] | ✅ | ❌ | ✅ | ❌ | ❌ | README is two bare shell commands; no description of what the project does, no tests, no history |
| [imgen] | ✅ | ✅ | ✅ | ❌ | ❌ | Missing test instructions and changelog |
| [imgui_gfx_example] | ✅ | ✅ | ❌ | ❌ | ❌ | Describes context but no build, test, or history sections |
| [libui-rs] _(fork)_ | ✅ | ✅ | ✅ | ✅ | ❌ | Fork of LeoTindall/libui-rs; tests note is only about compilation; no changelog for fork changes |
| [linux-init] | ✅ | ✅ | ❌ | ❌ | ❌ | Config-only repo; no setup, test, or history notes |
| [littlecat] | ✅ | ✅ | ❌ | ❌ | ❌ | Single-sentence description only; missing build, tests, history |
| [ludum-dare-47] | ⚠️ | ❌ | ❌ | ❌ | ❌ | README file exists but is completely empty (0 bytes) |
| [messd] | ✅ | ✅ | ❌ | ❌ | ❌ | One-sentence description; no build, test, or history |
| [opcode-decoder-generator] | ✅ | ✅ | ❌ | ❌ | ❌ | Good description and examples; no build, test, or history sections |
| [pydo-wrap] | ❌ | ❌ | ❌ | ❌ | ❌ | **No README file at all** (only LICENSE and .gitignore) |
| [raylib-py] _(fork)_ | ✅ | ✅ | ✅ | ✅ | ❌ | Fork of overdev/raylib-py; good coverage except changelog |
| [renderer] | ❌ | ❌ | ❌ | ❌ | ❌ | **No README file at all** |
| [rich] _(fork)_ | ✅ | ✅ | ✅ | ✅ | ❌ | Fork of Textualize/rich; no fork-specific changelog |
| [rust-search-engine-example] | ✅ | ✅ | ❌ | ❌ | ❌ | Marked work-in-progress; missing build, tests, history |
| [rust_simon] | ❌ | ❌ | ❌ | ❌ | ❌ | **No README file at all** |
| [sham] | ✅ | ❌ | ✅ | ✅ | ❌ | Dives straight into commands without describing what SHAM is; no history |
| [sham-sveltekit] | ✅ | ✅ | ✅ | ❌ | ❌ | Missing test instructions and changelog |
| [sham-vue] | ✅ | ✅ | ❌ | ❌ | ❌ | Single-line README; no build, test, or history |
| [sham_electron] | ✅ | ❌ | ✅ | ❌ | ❌ | Only shows start commands; no description, tests, or history |
| [shimlang] | ✅ | ✅ | ✅ | ❌ | ❌ | Missing test instructions and changelog (note: `LANGUAGE.md` exists) |
| [shimlang-builder] | ⚠️ | ❌ | ❌ | ❌ | ❌ | README contains only the project title |
| [shmboy] | ✅ | ✅ | ❌ | ❌ | ❌ | Good description and motivation; no build, test, or history |
| [shmip8] | ✅ | ✅ | ❌ | ❌ | ❌ | Good description and CHIP-8 background; no build, test, or history |
| [shmlox] | ✅ | ✅ | ❌ | ❌ | ❌ | Single-line README; no build, test, or history |
| [shmpylox] | ❌ | ❌ | ❌ | ❌ | ❌ | **No README file at all** (has `tests/` dir and `Makefile`) |
| [sous-vide-cooker] | ⚠️ | ❌ | ❌ | ❌ | ❌ | README contains only the project title (reStructuredText heading) |
| [SquareJam] | ✅ | ✅ | ❌ | ❌ | ✅ | How-to-play section; notes creation date; no source build instructions |
| [tex-template] | ✅ | ✅ | ✅ | ❌ | ❌ | Template repo; no tests or history expected but worth noting |
| [tiddly-wiki] | ⚠️ | ❌ | ❌ | ❌ | ❌ | README contains only the project title |
| [tinacms.org] _(fork)_ | ✅ | ✅ | ✅ | ❌ | ❌ | Fork of tinacms/tinacms.org; no test instructions or changelog |
| [UDKWiimote] | ✅ | ✅ | ✅ | ❌ | ✅ | Older project (2012); mentions VC++ build; no formal test instructions |
| [watch] | ✅ | ✅ | ❌ | ❌ | ❌ | Single-sentence description; no build, test, or history |
| [wimbledonlabs.github.io] | ✅ | ✅ | ❌ | ❌ | ❌ | Personal website repo; single-line description only |
| [zig_simon] | ❌ | ❌ | ❌ | ❌ | ❌ | **No README file at all** (has Makefile and raylib submodule) |

[basic-cpp-project-template]: https://github.com/shmolyneaux/basic-cpp-project-template
[bean_counter]: https://github.com/shmolyneaux/bean_counter
[bevy_simon]: https://github.com/shmolyneaux/bevy_simon
[clion_game]: https://github.com/shmolyneaux/clion_game
[hanzi-study-list-generator]: https://github.com/shmolyneaux/hanzi-study-list-generator
[imgen]: https://github.com/shmolyneaux/imgen
[imgui_gfx_example]: https://github.com/shmolyneaux/imgui_gfx_example
[libui-rs]: https://github.com/shmolyneaux/libui-rs
[linux-init]: https://github.com/shmolyneaux/linux-init
[littlecat]: https://github.com/shmolyneaux/littlecat
[ludum-dare-47]: https://github.com/shmolyneaux/ludum-dare-47
[messd]: https://github.com/shmolyneaux/messd
[opcode-decoder-generator]: https://github.com/shmolyneaux/opcode-decoder-generator
[pydo-wrap]: https://github.com/shmolyneaux/pydo-wrap
[raylib-py]: https://github.com/shmolyneaux/raylib-py
[renderer]: https://github.com/shmolyneaux/renderer
[rich]: https://github.com/shmolyneaux/rich
[rust-search-engine-example]: https://github.com/shmolyneaux/rust-search-engine-example
[rust_simon]: https://github.com/shmolyneaux/rust_simon
[sham]: https://github.com/shmolyneaux/sham
[sham-sveltekit]: https://github.com/shmolyneaux/sham-sveltekit
[sham-vue]: https://github.com/shmolyneaux/sham-vue
[sham_electron]: https://github.com/shmolyneaux/sham_electron
[shimlang]: https://github.com/shmolyneaux/shimlang
[shimlang-builder]: https://github.com/shmolyneaux/shimlang-builder
[shmboy]: https://github.com/shmolyneaux/shmboy
[shmip8]: https://github.com/shmolyneaux/shmip8
[shmlox]: https://github.com/shmolyneaux/shmlox
[shmpylox]: https://github.com/shmolyneaux/shmpylox
[sous-vide-cooker]: https://github.com/shmolyneaux/sous-vide-cooker
[SquareJam]: https://github.com/shmolyneaux/SquareJam
[tex-template]: https://github.com/shmolyneaux/tex-template
[tiddly-wiki]: https://github.com/shmolyneaux/tiddly-wiki
[tinacms.org]: https://github.com/shmolyneaux/tinacms.org
[UDKWiimote]: https://github.com/shmolyneaux/UDKWiimote
[watch]: https://github.com/shmolyneaux/watch
[wimbledonlabs.github.io]: https://github.com/shmolyneaux/wimbledonlabs.github.io
[zig_simon]: https://github.com/shmolyneaux/zig_simon

---

## Coverage Statistics

| Section | Repos with coverage | Out of 38 | % |
|---|:---:|:---:|:---:|
| README present & non-empty | 33 | 38 | 87% |
| Description | 26 | 38 | 68% |
| Build instructions | 16 | 38 | 42% |
| Test instructions | 5 | 38 | 13% |
| History / Changelog | 3 | 38 | 8% |

---

## Priority Improvements

### 🔴 Critical — No README at all (5 repos)

These repos have zero documentation. A README should be created from scratch.

| Repo | What exists | Suggested description |
|---|---|---|
| **pydo-wrap** | LICENSE, .gitignore | Add description of what pydo-wrap does, build/install steps |
| **renderer** | Cargo project (src/) | Add description of the renderer, how to build with `cargo build` |
| **rust_simon** | Cargo project + web assets | Add description of the Simon game, `cargo run` instructions |
| **shmpylox** | Makefile, lox.py, tests/ | Add description (Python Lox interpreter), `make` or `python lox.py` instructions |
| **zig_simon** | Makefile + raylib submodule | Add description of the Simon game, `make` build instructions |

### 🟠 High Priority — Effectively empty README (4 repos)

README exists but contains only a title or heading and nothing else.

| Repo | Current README | What to add |
|---|---|---|
| **ludum-dare-47** | Empty file (0 bytes) | Description of the Ludum Dare 47 entry, how to play/build, jam context |
| **shimlang-builder** | Only `# shimlang-builder` | What the builder does, how to run it, relationship to shimlang |
| **sous-vide-cooker** | Only title heading | What hardware/software is involved, setup instructions |
| **tiddly-wiki** | Only `# tiddly-wiki` | What customizations are tracked, how to set up |

### 🟡 Medium Priority — Missing description (5 repos)

README exists but lacks any explanation of what the project is.

| Repo | Missing | Suggested fix |
|---|---|---|
| **bevy_simon** | Description, Tests, History | Add a "What is this?" intro paragraph, test and history sections |
| **hanzi-study-list-generator** | Description, Tests, History | Explain what the tool generates and why; add expected output examples |
| **sham** | Description, History | Add a "What is SHAM?" intro; link to the other sham-* repos for context |
| **sham_electron** | Description, Tests, History | Describe the Electron-based SHAM client |

### 🟢 Lower Priority — Missing tests and/or history only

These repos have reasonably good descriptions and build docs but are missing
test and/or history sections. Test coverage is especially low across all repos
(only 5 of 38 have any test instructions at all).

**Missing tests:**
basic-cpp-project-template, bean_counter, bevy_simon, clion_game,
hanzi-study-list-generator, imgen, imgui_gfx_example, linux-init, littlecat,
messd, opcode-decoder-generator, sham-sveltekit, sham-vue, shimlang,
shmboy, shmip8, shmlox, tinacms.org, UDKWiimote, watch, wimbledonlabs.github.io

**Missing history/changelog:**
Nearly every repo (35 of 38). Even a brief "Development Notes" section with
dates/milestones would satisfy this criterion. Repos that do have some history
content: `SquareJam` (creation date + context), `UDKWiimote` (progress notes).

---

## Detection Methodology

Section presence was determined heuristically by scanning README content for:

- **Description**: any substantive prose in the opening section, or headings
  matching `About`, `Overview`, `What is`, `Introduction`, or project-name headings
  followed by descriptive text
- **Build**: headings or content matching `Build`, `Building`, `Install`,
  `Installation`, `Getting Started`, `Compiling`, `Prerequisites`, `Usage`,
  `Running`, `Development`, `Dev`, `Setup`; or code blocks containing build
  tool invocations (`cmake`, `cargo build/run`, `make`, `npm install/start/run`,
  `yarn`, `pip install`, `poetry install`, `zig build`)
- **Tests**: headings or content matching `Test`, `Testing`, `Tests`,
  `Running Tests`; or `make test`, `cargo test`, `pytest`, `npm test`
- **History**: headings or content matching `Changelog`, `History`,
  `Release Notes`, `Releases`, `Changes`, `What's New`, `Version`; or
  explicit dated version entries
