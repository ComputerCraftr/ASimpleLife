use std::path::{Path, PathBuf};

use crate::test_support::c::CSource;
use syn::visit::{self, Visit};
use tree_sitter::Node;

use crate::RequiredExt;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct InlineHashMixer {
    pub(super) path: PathBuf,
    pub(super) function: String,
}

#[derive(Default)]
struct RustMixerFacts {
    inline_xor_shift: bool,
    wrapping_multiply: bool,
}

impl<'ast> Visit<'ast> for RustMixerFacts {
    fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
        if matches!(expression.op, syn::BinOp::BitXor(_))
            && (contains_rust_shift(&expression.left) || contains_rust_shift(&expression.right))
        {
            self.inline_xor_shift = true;
        }
        visit::visit_expr_binary(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        self.wrapping_multiply |= expression.method == "wrapping_mul";
        visit::visit_expr_method_call(self, expression);
    }
}

fn contains_rust_shift(expression: &syn::Expr) -> bool {
    struct ShiftVisitor(bool);
    impl<'ast> Visit<'ast> for ShiftVisitor {
        fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
            self.0 |= matches!(expression.op, syn::BinOp::Shl(_) | syn::BinOp::Shr(_));
            visit::visit_expr_binary(self, expression);
        }
    }
    let mut visitor = ShiftVisitor(false);
    visitor.visit_expr(expression);
    visitor.0
}

pub(super) fn rust_inline_hash_mixers(path: &Path, source: &str) -> Vec<InlineHashMixer> {
    struct Collector<'a> {
        path: &'a Path,
        mixers: Vec<InlineHashMixer>,
    }
    impl<'ast> Visit<'ast> for Collector<'_> {
        fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
            self.inspect(function.sig.ident.to_string(), &function.block);
        }

        fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
            self.inspect(function.sig.ident.to_string(), &function.block);
        }
    }
    impl Collector<'_> {
        fn inspect(&mut self, function: String, body: &syn::Block) {
            let mut facts = RustMixerFacts::default();
            facts.visit_block(body);
            if facts.inline_xor_shift && facts.wrapping_multiply {
                self.mixers.push(InlineHashMixer {
                    path: self.path.to_path_buf(),
                    function,
                });
            }
        }
    }

    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust source {}: {error}", path.display())
    });
    let mut collector = Collector {
        path,
        mixers: Vec::new(),
    };
    collector.visit_file(&file);
    collector.mixers
}

pub(super) fn c_inline_hash_mixers(path: &Path, source: &str) -> Vec<InlineHashMixer> {
    let syntax = CSource::try_parse(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse C source {}: {error}", path.display())
    });
    let mut mixers = Vec::new();
    let mut pending = vec![syntax.root()];
    while let Some(node) = pending.pop() {
        if node.kind() == "function_definition" {
            inspect_c_function(node, source.as_bytes(), path, &mut mixers);
            continue;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    mixers
}

fn inspect_c_function(
    function: Node<'_>,
    source: &[u8],
    path: &Path,
    mixers: &mut Vec<InlineHashMixer>,
) {
    let declarator = function
        .child_by_field_name("declarator")
        .or_invariant("C function should have a declarator");
    let name = first_c_identifier(declarator, source).or_invariant("C function should have a name");
    let body = function
        .child_by_field_name("body")
        .or_invariant("C function should have a body");
    let mut has_inline_xor_shift = false;
    let mut has_multiply = false;
    let mut pending = vec![body];
    while let Some(node) = pending.pop() {
        if node.kind() == "binary_expression" {
            let operator = c_binary_operator(node, source);
            has_multiply |= operator == Some("*");
            has_inline_xor_shift |=
                operator == Some("^") && c_expression_contains_shift(node, source);
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    if has_inline_xor_shift && has_multiply {
        mixers.push(InlineHashMixer {
            path: path.to_path_buf(),
            function: name,
        });
    }
}

fn c_expression_contains_shift(node: Node<'_>, source: &[u8]) -> bool {
    let mut pending = Vec::new();
    let mut cursor = node.walk();
    pending.extend(node.named_children(&mut cursor));
    while let Some(child) = pending.pop() {
        if child.kind() == "binary_expression"
            && matches!(c_binary_operator(child, source), Some("<<" | ">>"))
        {
            return true;
        }
        let mut cursor = child.walk();
        pending.extend(child.named_children(&mut cursor));
    }
    false
}

fn c_binary_operator<'a>(node: Node<'_>, source: &'a [u8]) -> Option<&'a str> {
    let left = node.child_by_field_name("left")?;
    let right = node.child_by_field_name("right")?;
    std::str::from_utf8(&source[left.end_byte()..right.start_byte()])
        .ok()
        .map(str::trim)
}

fn first_c_identifier(node: Node<'_>, source: &[u8]) -> Option<String> {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if node.kind() == "identifier" {
            return node.utf8_text(source).ok().map(str::to_owned);
        }
        if let Some(declarator) = node.child_by_field_name("declarator") {
            pending.push(declarator);
            continue;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    None
}
