use crate::RequiredExt;
use tree_sitter::{Node, Parser, Tree};

/// Queries and edits are anchored to C syntax nodes, never comments or arbitrary
/// byte matches. The C compiler remains the authority for preprocessing/types.
pub(crate) struct CSource<'a> {
    source: &'a str,
    tree: Tree,
}

impl<'a> CSource<'a> {
    pub(crate) fn parse(source: &'a str) -> Self {
        Self::try_parse(source).or_invariant("valid C syntax without parser recovery")
    }

    pub(crate) fn try_parse(source: &'a str) -> Result<Self, String> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .or_invariant("C grammar");
        let tree = parser.parse(source, None).or_invariant("C syntax tree");
        let parsed = Self { source, tree };
        let errors = parsed.errors();
        if errors.is_empty() {
            Ok(parsed)
        } else {
            Err(errors.join("\n"))
        }
    }

    fn nodes(&self) -> Vec<Node<'_>> {
        descendants(self.tree.root_node())
    }

    pub(crate) fn root(&self) -> Node<'_> {
        self.tree.root_node()
    }

    fn text(&self, node: Node<'_>) -> &str {
        &self.source[node.byte_range()]
    }

    pub(crate) fn has_identifier(&self, name: &str) -> bool {
        self.nodes()
            .iter()
            .any(|&node| is_identifier(node) && self.text(node) == name)
    }

    pub(crate) fn has_identifier_prefix(&self, prefix: &str) -> bool {
        self.nodes()
            .iter()
            .any(|&node| is_identifier(node) && self.text(node).starts_with(prefix))
    }

    pub(crate) fn declares(&self, name: &str) -> bool {
        self.nodes().iter().any(|node| {
            node.kind() == "init_declarator"
                && node
                    .child_by_field_name("declarator")
                    .is_some_and(|id| self.text(id) == name)
        })
    }

    pub(crate) fn errors(&self) -> Vec<String> {
        self.nodes()
            .into_iter()
            .filter(|node| node.is_error() || node.is_missing())
            .map(|node| {
                format!(
                    "{} at {:?}: {}",
                    node.kind(),
                    node.start_position(),
                    self.text(node)
                )
            })
            .collect()
    }

    pub(crate) fn unconditional_errors(&self) -> Vec<&str> {
        self.nodes()
            .into_iter()
            .filter(|node| {
                node.kind() == "preproc_call"
                    && node
                        .parent()
                        .is_some_and(|parent| parent.kind() == "translation_unit")
                    && node
                        .child_by_field_name("directive")
                        .is_some_and(|directive| self.text(directive) == "#error")
            })
            .filter_map(|node| {
                node.child_by_field_name("argument")
                    .map(|argument| self.text(argument).trim())
            })
            .collect()
    }

    pub(crate) fn has_call(&self, name: &str) -> bool {
        self.nodes().iter().any(|&node| {
            node.kind() == "call_expression"
                && node
                    .child_by_field_name("function")
                    .is_some_and(|function| {
                        function.kind() == "identifier" && self.text(function) == name
                    })
        })
    }

    pub(crate) fn has_call_using_identifier_prefix(
        &self,
        name: &str,
        argument: u32,
        prefix: &str,
    ) -> bool {
        self.nodes().iter().any(|&node| {
            node.kind() == "call_expression"
                && node
                    .child_by_field_name("function")
                    .is_some_and(|f| self.text(f) == name)
                && node
                    .child_by_field_name("arguments")
                    .and_then(|args| args.named_child(argument))
                    .is_some_and(|arg| {
                        arg.kind() == "identifier" && self.text(arg).starts_with(prefix)
                    })
        })
    }

    pub(crate) fn has_syntax(&self, snippet: &str) -> bool {
        let wrapped = format!("void query(void) {{ {snippet}; }}");
        let query = CSource::parse(&wrapped);
        assert!(
            !query.tree.root_node().has_error(),
            "invalid C query: {snippet}"
        );
        let body = query
            .nodes()
            .into_iter()
            .find(|node| node.kind() == "compound_statement")
            .or_invariant("query body");
        let mut cursor = body.walk();
        let node = body
            .named_children(&mut cursor)
            .find(|node| node.kind() != "comment")
            .or_invariant("query statement");
        let node = if node.kind() == "expression_statement" && !snippet.trim_end().ends_with(';') {
            node.named_child(0).or_invariant("query expression")
        } else {
            node
        };
        let expected = signature(node, query.source);
        self.nodes().into_iter().any(|candidate| {
            candidate.kind() == node.kind()
                && !candidate.has_error()
                && signature(candidate, self.source) == expected
        })
    }

    pub(crate) fn has_condition(&self, statement_kind: &str, expression: &str) -> bool {
        let wrapped = format!("void query(void) {{ if ({expression}) {{}} }}");
        let query = CSource::parse(&wrapped);
        let condition = query
            .nodes()
            .into_iter()
            .find(|node| node.kind() == "if_statement")
            .and_then(|node| node.child_by_field_name("condition"))
            .or_invariant("condition query");
        let expected = signature(condition, &wrapped);
        self.nodes()
            .into_iter()
            .filter(|node| node.kind() == statement_kind)
            .any(|node| {
                node.child_by_field_name("condition")
                    .is_some_and(|condition| signature(condition, self.source) == expected)
            })
    }

    pub(crate) fn define_values(&self, name: &str) -> Vec<&str> {
        self.nodes()
            .into_iter()
            .filter(|node| is_unconditional_define(*node))
            .filter(|node| {
                node.child_by_field_name("name")
                    .is_some_and(|n| self.text(n) == name)
            })
            .filter_map(|node| {
                node.child_by_field_name("value")
                    .map(|n| self.text(n).trim())
            })
            .collect()
    }

    pub(crate) fn replace_define(&self, name: &str, value: &str) -> Result<String, String> {
        let ranges: Vec<_> = self
            .nodes()
            .into_iter()
            .filter(|node| is_unconditional_define(*node))
            .filter(|node| {
                node.child_by_field_name("name")
                    .is_some_and(|n| self.text(n) == name)
            })
            .filter_map(|node| node.child_by_field_name("value").map(|n| n.byte_range()))
            .collect();
        self.replace_unique(ranges, value)
    }

    /// Replace the complete statement of one direct call in main, not a
    /// declaration, a comment, or a call in an unrelated helper.
    pub(crate) fn replace_main_call(
        &self,
        name: &str,
        replacement: &str,
    ) -> Result<String, String> {
        let ranges = self
            .nodes()
            .into_iter()
            .filter(|node| node.kind() == "function_definition")
            .filter(|node| {
                node.child_by_field_name("declarator")
                    .is_some_and(|declarator| {
                        declarator
                            .child_by_field_name("declarator")
                            .is_some_and(|id| self.text(id) == "main")
                    })
            })
            .flat_map(descendants)
            .filter(|node| node.kind() == "expression_statement")
            .filter(|node| {
                node.named_child(0).is_some_and(|call| {
                    call.kind() == "call_expression"
                        && call
                            .child_by_field_name("function")
                            .is_some_and(|f| self.text(f) == name)
                })
            })
            .map(|node| node.byte_range())
            .collect();
        self.replace_unique(ranges, replacement)
    }

    fn replace_unique(
        &self,
        ranges: Vec<std::ops::Range<usize>>,
        replacement: &str,
    ) -> Result<String, String> {
        if ranges.len() != 1 {
            return Err(format!(
                "C AST edit requires one match, found {}",
                ranges.len()
            ));
        }
        let mut output = self.source.to_owned();
        output.replace_range(ranges[0].clone(), replacement);
        Ok(output)
    }
}

fn is_identifier(node: Node<'_>) -> bool {
    matches!(
        node.kind(),
        "identifier" | "type_identifier" | "field_identifier"
    )
}

fn is_unconditional_define(node: Node<'_>) -> bool {
    node.kind() == "preproc_def"
        && node
            .parent()
            .is_some_and(|parent| parent.kind() == "translation_unit")
}

fn descendants(root: Node<'_>) -> Vec<Node<'_>> {
    let mut pending = vec![root];
    let mut result = Vec::new();
    while let Some(node) = pending.pop() {
        if node.kind() == "comment" {
            continue;
        }
        result.push(node);
        pending.extend(
            (0..node.child_count())
                .rev()
                .filter_map(|index| node.child(index)),
        );
    }
    result
}

fn signature<'a>(node: Node<'_>, source: &'a str) -> Vec<(u16, &'a str)> {
    descendants(node)
        .into_iter()
        .map(|node| {
            (
                node.kind_id(),
                if node.child_count() == 0 {
                    &source[node.byte_range()]
                } else {
                    ""
                },
            )
        })
        .collect()
}

#[test]
fn c_queries_ignore_comments_literals_and_formatting() {
    let source = CSource::parse(
        r#"
        // bf_missing();
        const char *text = "bf_missing();";
        int main(void) { ptr /* gap */ = wrap ( ptr, 2, LEN ); stats (); }
    "#,
    );
    assert!(!source.has_call("bf_missing"));
    assert!(!source.has_identifier("bf_missing"));
    assert!(source.has_syntax("ptr = wrap(ptr, 2, LEN);"));
    assert!(!source.has_syntax("ptr = wrap(ptr, 3, LEN);"));
    let conditional = CSource::parse("void f(void) { if (tape[ptr] != 0) { diverge(); } }");
    assert!(!conditional.has_condition("while_statement", "tape[ptr] != 0"));
    assert!(conditional.has_condition("if_statement", "tape[ptr] != 0"));
    assert!(
        source
            .replace_main_call("stats", "dump(); stats();")
            .is_ok()
    );
}

#[test]
fn c_edits_reject_absent_ambiguous_or_unrelated_calls() {
    for source in [
        "void f(void) { stats(); }",
        "int main(void) { stats(); stats(); }",
    ] {
        assert!(
            CSource::parse(source)
                .replace_main_call("stats", "dump();")
                .is_err(),
            "{source}"
        );
    }
    let source = CSource::parse("// #define CAP 9\n#define CAP 1\nint main(void) {}\n");
    assert_eq!(source.define_values("CAP"), ["1"]);
    assert_eq!(
        source
            .replace_define("CAP", "2")
            .or_invariant("define edit"),
        "// #define CAP 9\n#define CAP 2\nint main(void) {}\n"
    );
}

#[test]
fn c_queries_reject_recovered_or_missing_syntax_instead_of_passing_absence_checks() {
    for source in ["int main(void) { stats(", "int main(void) { return 0;"] {
        assert!(
            CSource::try_parse(source).is_err(),
            "malformed C accepted: {source}"
        );
    }
    let source = CSource::parse("void bf_poly(void); int main(void) { other(bf_poly_lhs); }");
    assert!(!source.has_call_using_identifier_prefix("bf_poly", 0, "bf_poly_lhs"));
}

#[test]
fn c_configuration_edits_select_unconditional_definitions_not_template_defaults() {
    let source = "#define CAP 4096\n#ifndef CAP\n#define CAP 64\n#endif\n";
    let syntax = CSource::parse(source);
    assert_eq!(syntax.define_values("CAP"), ["4096"]);
    assert_eq!(
        syntax
            .replace_define("CAP", "16")
            .or_invariant("configuration edit"),
        "#define CAP 16\n#ifndef CAP\n#define CAP 64\n#endif\n"
    );
    assert!(
        CSource::parse("#define CAP 1\n#define CAP 2\n")
            .replace_define("CAP", "3")
            .is_err()
    );
    let source =
        CSource::parse("#if WIDTH > 63\n#error invalid width\n#endif\n#error invalid program\n");
    assert_eq!(source.unconditional_errors(), ["invalid program"]);
}
