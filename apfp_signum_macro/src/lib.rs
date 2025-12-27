use proc_macro::TokenStream;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Expr, ExprBinary, ExprCall, ExprField, ExprGroup, ExprLit, ExprParen,
    ExprPath, ExprUnary, UnOp,
};

#[proc_macro]
pub fn apfp_signum(input: TokenStream) -> TokenStream {
    let expr = parse_macro_input!(input as Expr);
    let crate_path = match crate_name("apfp") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        Err(_) => quote!(::apfp),
    };
    let mut builder = Builder::default();
    match expand_expr(&expr, &crate_path, &mut builder) {
        Ok(expanded) => {
            let ExpandedExpr {
                expr: expanded_expr,
                ty: expr_type,
                value,
                abs,
                stmts,
                op_count,
            } = expanded;
            let gamma = gamma_from_ops(op_count);
            let mut dd_builder = Builder::default();
            let dd_expanded = match expand_dd_expr(&expr, &mut dd_builder) {
                Ok(tokens) => tokens,
                Err(err) => return err.to_compile_error().into(),
            };
            TokenStream::from(quote! {
                {
                    use #crate_path::analysis::adaptive_signum::{
                        dd_add, dd_from, dd_mul, dd_neg, dd_signum, dd_square, dd_sub, Scalar,
                        signum_exact, square, Dd, Signum,
                    };
                    #stmts
                    let _apfp_value: f64 = #value;
                    let _apfp_abs: f64 = #abs;
                    let _apfp_err: f64 = #gamma * _apfp_abs;
                    if _apfp_value > _apfp_err {
                        1
                    } else if _apfp_value < -_apfp_err {
                        -1
                    } else {
                        #dd_expanded
                        if let Some(sign) = dd_signum(_apfp_dd_value) {
                            sign
                        } else {
                            type ExprType = #expr_type;
                            let expr: ExprType = #expanded_expr;
                            let mut buffer =
                                core::mem::MaybeUninit::<[f64; <ExprType as Signum>::STACK_LEN]>::uninit();
                            let buffer = unsafe { &mut *buffer.as_mut_ptr() };
                            signum_exact(expr, buffer)
                        }
                    }
                }
            })
        }
        Err(err) => err.to_compile_error().into(),
    }
}

#[derive(Default)]
struct Builder {
    counter: usize,
}

struct ExpandedExpr {
    expr: proc_macro2::TokenStream,
    ty: proc_macro2::TokenStream,
    value: proc_macro2::TokenStream,
    abs: proc_macro2::TokenStream,
    stmts: proc_macro2::TokenStream,
    op_count: usize,
}

fn expand_dd_expr(
    expr: &Expr,
    builder: &mut Builder,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let expanded = expand_dd_expr_inner(expr, builder)?;
    let stmts = expanded.stmts;
    let value = expanded.value;
    Ok(quote! {
        #stmts
        let _apfp_dd_value: Dd = #value;
    })
}

struct ExpandedDd {
    value: proc_macro2::TokenStream,
    stmts: proc_macro2::TokenStream,
}

fn expand_dd_expr_inner(expr: &Expr, builder: &mut Builder) -> Result<ExpandedDd, syn::Error> {
    match expr {
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let lhs = expand_dd_expr_inner(left, builder)?;
            let rhs = expand_dd_expr_inner(right, builder)?;
            let lhs_value = lhs.value.clone();
            let rhs_value = rhs.value.clone();
            let lhs_stmts = lhs.stmts.clone();
            let rhs_stmts = rhs.stmts.clone();
            let value_ident = builder.next_ident("_apfp_dd");
            let op_call = match op {
                syn::BinOp::Add(_) => quote! { dd_add(#lhs_value, #rhs_value) },
                syn::BinOp::Sub(_) => quote! { dd_sub(#lhs_value, #rhs_value) },
                syn::BinOp::Mul(_) => quote! { dd_mul(#lhs_value, #rhs_value) },
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "unsupported operator; use +, -, or *",
                    ));
                }
            };
            Ok(ExpandedDd {
                value: quote! { #value_ident },
                stmts: quote! {
                    #lhs_stmts
                    #rhs_stmts
                    let #value_ident: Dd = #op_call;
                },
            })
        }
        Expr::Unary(ExprUnary { op: UnOp::Neg(_), expr, .. }) => {
            let inner = expand_dd_expr_inner(expr, builder)?;
            let value_ident = builder.next_ident("_apfp_dd");
            let inner_value = inner.value.clone();
            let inner_stmts = inner.stmts.clone();
            Ok(ExpandedDd {
                value: quote! { #value_ident },
                stmts: quote! {
                    #inner_stmts
                    let #value_ident: Dd = dd_neg(#inner_value);
                },
            })
        }
        Expr::Paren(ExprParen { expr, .. }) => expand_dd_expr_inner(expr, builder),
        Expr::Group(ExprGroup { expr, .. }) => expand_dd_expr_inner(expr, builder),
        Expr::Call(ExprCall { func, args, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = &**func {
                if path.is_ident("square") && args.len() == 1 {
                    let inner = expand_dd_expr_inner(&args[0], builder)?;
                    let value_ident = builder.next_ident("_apfp_dd");
                    let inner_value = inner.value.clone();
                    let inner_stmts = inner.stmts.clone();
                    return Ok(ExpandedDd {
                        value: quote! { #value_ident },
                        stmts: quote! {
                            #inner_stmts
                            let #value_ident: Dd = dd_square(#inner_value);
                        },
                    });
                }
            }
            Err(syn::Error::new_spanned(
                expr,
                "unsupported function call; only square(x) is supported",
            ))
        }
        Expr::Lit(ExprLit { .. }) | Expr::Path(ExprPath { .. }) | Expr::Field(ExprField { .. }) => {
            let value_ident = builder.next_ident("_apfp_dd");
            Ok(ExpandedDd {
                value: quote! { #value_ident },
                stmts: quote! {
                    let #value_ident: Dd = dd_from((#expr) as f64);
                },
            })
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "unsupported expression; use literals, identifiers, +, -, *, unary -, or square(x)",
        )),
    }
}

fn gamma_from_ops(op_count: usize) -> proc_macro2::TokenStream {
    if op_count == 0 {
        return quote!(0.0_f64);
    }
    let u = 0.5 * f64::EPSILON;
    let n = op_count as f64;
    let gamma = (n * u) / (1.0 - n * u);
    let lit = proc_macro2::Literal::f64_suffixed(gamma);
    quote!(#lit)
}

impl Builder {
    fn next_ident(&mut self, prefix: &str) -> proc_macro2::Ident {
        let ident = format_ident!("{}_{}", prefix, self.counter);
        self.counter += 1;
        ident
    }
}

fn expand_expr(
    expr: &Expr,
    crate_path: &proc_macro2::TokenStream,
    builder: &mut Builder,
) -> Result<ExpandedExpr, syn::Error> {
    match expr {
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let lhs = expand_expr(left, crate_path, builder)?;
            let rhs = expand_expr(right, crate_path, builder)?;
            let lhs_expr = lhs.expr.clone();
            let rhs_expr = rhs.expr.clone();
            let lhs_ty = lhs.ty.clone();
            let rhs_ty = rhs.ty.clone();
            let lhs_value = lhs.value.clone();
            let rhs_value = rhs.value.clone();
            let lhs_abs = lhs.abs.clone();
            let rhs_abs = rhs.abs.clone();
            let lhs_stmts = lhs.stmts.clone();
            let rhs_stmts = rhs.stmts.clone();
            let value_ident = builder.next_ident("_apfp_v");
            let abs_ident = builder.next_ident("_apfp_abs");
            let (out_expr, out_ty) = match op {
                syn::BinOp::Add(_) => (
                    quote! { (#lhs_expr) + (#rhs_expr) },
                    quote! { #crate_path::analysis::adaptive_signum::Sum<#lhs_ty, #rhs_ty> },
                ),
                syn::BinOp::Sub(_) => (
                    quote! { (#lhs_expr) - (#rhs_expr) },
                    quote! { #crate_path::analysis::adaptive_signum::Diff<#lhs_ty, #rhs_ty> },
                ),
                syn::BinOp::Mul(_) => (
                    quote! { (#lhs_expr) * (#rhs_expr) },
                    quote! { #crate_path::analysis::adaptive_signum::Product<#lhs_ty, #rhs_ty> },
                ),
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "unsupported operator; use +, -, or *",
                    ));
                }
            };
            let fast_stmt = match op {
                syn::BinOp::Add(_) => quote! {
                    let #value_ident: f64 = #lhs_value + #rhs_value;
                    let #abs_ident: f64 = #lhs_abs + #rhs_abs;
                },
                syn::BinOp::Sub(_) => quote! {
                    let #value_ident: f64 = #lhs_value - #rhs_value;
                    let #abs_ident: f64 = #lhs_abs + #rhs_abs;
                },
                syn::BinOp::Mul(_) => quote! {
                    let #value_ident: f64 = #lhs_value * #rhs_value;
                    let #abs_ident: f64 = #value_ident.abs();
                },
                _ => unreachable!("checked above"),
            };
            Ok(ExpandedExpr {
                expr: out_expr,
                ty: out_ty,
                value: quote! { #value_ident },
                abs: quote! { #abs_ident },
                stmts: quote! {
                    #lhs_stmts
                    #rhs_stmts
                    #fast_stmt
                },
                op_count: lhs.op_count + rhs.op_count + 1,
            })
        }
        Expr::Unary(ExprUnary { op: UnOp::Neg(_), expr, .. }) => {
            let inner = expand_expr(expr, crate_path, builder)?;
            let value_ident = builder.next_ident("_apfp_v");
            let abs_ident = builder.next_ident("_apfp_abs");
            let inner_expr = inner.expr.clone();
            let inner_ty = inner.ty.clone();
            let inner_value = inner.value.clone();
            let inner_abs = inner.abs.clone();
            let inner_stmts = inner.stmts.clone();
            Ok(ExpandedExpr {
                expr: quote! { -(#inner_expr) },
                ty: quote! { #crate_path::analysis::adaptive_signum::Negate<#inner_ty> },
                value: quote! { #value_ident },
                abs: quote! { #abs_ident },
                stmts: quote! {
                    #inner_stmts
                    let #value_ident: f64 = -#inner_value;
                    let #abs_ident: f64 = #inner_abs;
                },
                op_count: inner.op_count,
            })
        }
        Expr::Paren(ExprParen { expr, .. }) => expand_expr(expr, crate_path, builder),
        Expr::Group(ExprGroup { expr, .. }) => expand_expr(expr, crate_path, builder),
        Expr::Call(ExprCall { func, args, .. }) => {
            if let Expr::Path(ExprPath { path, .. }) = &**func {
                if path.is_ident("square") && args.len() == 1 {
                    let inner = expand_expr(&args[0], crate_path, builder)?;
                    let value_ident = builder.next_ident("_apfp_v");
                    let abs_ident = builder.next_ident("_apfp_abs");
                    let inner_expr = inner.expr.clone();
                    let inner_ty = inner.ty.clone();
                    let inner_value = inner.value.clone();
                    let inner_stmts = inner.stmts.clone();
                    return Ok(ExpandedExpr {
                        expr: quote! { square(#inner_expr) },
                        ty: quote! { #crate_path::analysis::adaptive_signum::Square<#inner_ty> },
                        value: quote! { #value_ident },
                        abs: quote! { #abs_ident },
                        stmts: quote! {
                            #inner_stmts
                            let #value_ident: f64 = #inner_value * #inner_value;
                            let #abs_ident: f64 = #value_ident.abs();
                        },
                        op_count: inner.op_count + 1,
                    });
                }
            }
            Err(syn::Error::new_spanned(
                expr,
                "unsupported function call; only square(x) is supported",
            ))
        }
        Expr::Lit(ExprLit { .. }) | Expr::Path(ExprPath { .. }) | Expr::Field(ExprField { .. }) => {
            let value_ident = builder.next_ident("_apfp_v");
            let abs_ident = builder.next_ident("_apfp_abs");
            Ok(ExpandedExpr {
                expr: quote! { Scalar(#expr) },
                ty: quote! { #crate_path::analysis::adaptive_signum::Scalar },
                value: quote! { #value_ident },
                abs: quote! { #abs_ident },
                stmts: quote! {
                    let #value_ident: f64 = (#expr) as f64;
                    let #abs_ident: f64 = #value_ident.abs();
                },
                op_count: 0,
            })
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "unsupported expression; use literals, identifiers, +, -, *, unary -, or square(x)",
        )),
    }
}
