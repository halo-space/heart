//! Compile-time validation and adapters for Workflow Nodes and Tools.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote, quote_spanned};
use syn::{
    FnArg, GenericArgument, Ident, ImplItem, Item, ItemFn, ItemImpl, LitStr, PathArguments,
    ReturnType, Signature, Type,
    parse::{Parse, ParseStream},
    spanned::Spanned,
    visit::{self, Visit},
};

enum Mode {
    Function,
    Call(Ident),
    Signal(Ident),
}

impl Parse for Mode {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::Function);
        }
        let key: Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let method: Ident = input.parse()?;
        if !input.is_empty() {
            return Err(input.error("use exactly one of call = method or signal = method"));
        }
        match key.to_string().as_str() {
            "call" => Ok(Self::Call(method)),
            "signal" => Ok(Self::Signal(method)),
            _ => Err(syn::Error::new(
                key.span(),
                "expected call = method or signal = method",
            )),
        }
    }
}

#[proc_macro_attribute]
pub fn node(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let result = syn::parse::<Mode>(attributes).and_then(|mode| {
        syn::parse::<Item>(item).and_then(|item| {
            let root = root()?;
            expand(mode, item, &root)
        })
    });
    result
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}

struct ToolAttributes {
    name: Option<LitStr>,
    description: Option<LitStr>,
}

impl Parse for ToolAttributes {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            let value: LitStr = input.parse()?;
            match key.to_string().as_str() {
                "name" if name.is_none() => name = Some(value),
                "description" if description.is_none() => description = Some(value),
                "name" => {
                    return Err(syn::Error::new(key.span(), "duplicate tool name"));
                }
                "description" => {
                    return Err(syn::Error::new(key.span(), "duplicate tool description"));
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected name = \"...\" or description = \"...\"",
                    ));
                }
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            } else if !input.is_empty() {
                return Err(input.error("separate tool attributes with commas"));
            }
        }
        let Some(description) = description else {
            return Err(input.error("#[tool] requires description = \"...\""));
        };
        if description.value().trim().is_empty() {
            return Err(syn::Error::new(
                description.span(),
                "tool description must not be empty",
            ));
        }
        Ok(Self {
            name,
            description: Some(description),
        })
    }
}

#[proc_macro_attribute]
pub fn tool(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let result = syn::parse::<ToolAttributes>(attributes).and_then(|attributes| {
        syn::parse::<ItemFn>(item).and_then(|function| {
            let root = root()?;
            expand_tool(attributes, function, &root)
        })
    });
    result
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}

fn root() -> syn::Result<Tokens> {
    use proc_macro_crate::{FoundCrate, crate_name};
    match crate_name("halo-agent") {
        Ok(FoundCrate::Itself) => Ok(quote!(::halo)),
        // Cargo exposes this package's explicit `[lib] name = "halo"` when
        // the dependency has not been renamed by its consumer.
        Ok(FoundCrate::Name(name)) if name == "halo_agent" => Ok(quote!(::halo)),
        Ok(FoundCrate::Name(name)) => {
            let name = syn::parse_str::<Ident>(&name)
                .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))?;
            Ok(quote!(::#name))
        }
        Err(error) => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            error.to_string(),
        )),
    }
}

struct ToolIo {
    output: Type,
    error: Type,
}

fn expand_tool(
    attributes: ToolAttributes,
    mut function: ItemFn,
    root: &Tokens,
) -> syn::Result<Tokens> {
    if function.sig.asyncness.is_none() {
        return Err(syn::Error::new(
            function.sig.span(),
            "#[tool] requires an async function",
        ));
    }
    if function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || function.sig.variadic.is_some()
    {
        return Err(syn::Error::new(
            function.sig.span(),
            "use a regular safe Rust async function for #[tool]",
        ));
    }
    if !function.sig.generics.params.is_empty() || function.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            function.sig.generics.span(),
            "generic tool functions require an explicit Tool implementation",
        ));
    }

    let Some(FnArg::Typed(cancellation)) = function.sig.inputs.last() else {
        return Err(syn::Error::new(
            function.sig.inputs.span(),
            "#[tool] requires a final &Cancellation parameter",
        ));
    };
    if !matches!(&*cancellation.ty, Type::Reference(reference)
        if reference.mutability.is_none() && named(&reference.elem, "Cancellation"))
    {
        return Err(syn::Error::new(
            cancellation.ty.span(),
            "the final #[tool] parameter must be &Cancellation",
        ));
    }
    if function.sig.inputs.len() == 1 {
        return Err(syn::Error::new(
            function.sig.inputs.span(),
            "#[tool] requires at least one typed business argument before &Cancellation",
        ));
    }

    let mut arguments = Vec::new();
    for argument in function
        .sig
        .inputs
        .iter()
        .take(function.sig.inputs.len() - 1)
    {
        let FnArg::Typed(argument) = argument else {
            return Err(syn::Error::new(
                argument.span(),
                "#[tool] only supports typed function arguments",
            ));
        };
        let syn::Pat::Ident(pattern) = &*argument.pat else {
            return Err(syn::Error::new(
                argument.pat.span(),
                "#[tool] arguments must use simple identifiers",
            ));
        };
        owned(&argument.ty).map_err(|error| {
            syn::Error::new(
                error.span(),
                "#[tool] arguments must be owned types without lifetimes",
            )
        })?;
        arguments.push((pattern.ident.clone(), (*argument.ty).clone()));
    }

    let ToolIo { output, error } = tool_result_types(&function.sig.output)?;
    owned(&output).map_err(|error| {
        syn::Error::new(
            error.span(),
            "#[tool] output must be an owned type without lifetimes",
        )
    })?;
    owned(&error).map_err(|error| {
        syn::Error::new(
            error.span(),
            "#[tool] error must be an owned type without lifetimes",
        )
    })?;

    let original_name = function.sig.ident.clone();
    let implementation_name = format_ident!("__heart_tool_{}_implementation", original_name);
    function.sig.ident = implementation_name.clone();
    let tool_type = format_ident!("__HeartTool{}", pascal_case(&original_name.to_string()));
    let constructor = original_name;
    let visibility = function.vis.clone();
    function.vis = syn::Visibility::Inherited;
    let tool_name = attributes
        .name
        .unwrap_or_else(|| LitStr::new(&constructor.to_string(), constructor.span()));
    let description = attributes.description.expect("description is required");
    let field_definitions = arguments.iter().map(|(name, ty)| quote!(#name: #ty));
    let argument_names = arguments.iter().map(|(name, _)| quote!(parsed.#name));
    let args_type = format_ident!(
        "__HeartTool{}Arguments",
        pascal_case(&constructor.to_string())
    );
    let output_message = if named(&output, "Message") {
        quote!(output)
    } else {
        quote! {
            let output = #root::__tool::serde_json::to_value(output).map_err(|error| {
                #root::Error::new("SERIALIZATION", "tool output could not be serialized")
                    .with_details(error.to_string())
            })?;
            #root::agent::Message::function(output)
        }
    };
    let call = quote! {
        let output = #implementation_name(#(#argument_names,)* cancellation)
            .await
            .map_err(::core::convert::Into::into)?;
        let message = { #output_message };
        Ok(message)
    };
    let schema = quote! {
        let schema = #root::__tool::serde_json::to_value(
            #root::__tool::schemars::schema_for!(#args_type),
        )
        .expect("tool arguments schema must serialize");
        let parameters = schema
            .as_object()
            .cloned()
            .expect("tool arguments schema must be an object");
    };
    Ok(quote! {
        #function

        #[derive(#root::__tool::serde::Deserialize, #root::__tool::schemars::JsonSchema)]
        #[allow(non_camel_case_types)]
        #visibility struct #args_type {
            #(#field_definitions,)*
        }

        #[allow(non_camel_case_types)]
        #visibility struct #tool_type;

        #visibility fn #constructor() -> #tool_type {
            #tool_type
        }

        impl #root::agent::tool::Tool for #tool_type {
            fn definition(&self) -> &#root::agent::tool::ToolDefinition {
                static DEFINITION: ::std::sync::OnceLock<#root::agent::tool::ToolDefinition> =
                    ::std::sync::OnceLock::new();
                DEFINITION.get_or_init(|| {
                    #schema
                    #root::agent::tool::ToolDefinition {
                        name: #tool_name.to_owned(),
                        description: #description.to_owned(),
                        parameters,
                    }
                })
            }

            async fn call(
                &self,
                call: &#root::agent::tool::Call,
                cancellation: &#root::Cancellation,
            ) -> ::std::result::Result<#root::agent::Message, #root::Error> {
                let parsed: #args_type = #root::__tool::serde_json::from_value(
                    #root::__tool::serde_json::Value::Object(call.arguments.clone()),
                )
                .map_err(|error| {
                    #root::Error::new("INVALID_ARGUMENTS", "tool arguments could not be decoded")
                        .with_details(error.to_string())
                })?;
                #call
            }
        }
    })
}

fn tool_result_types(output: &ReturnType) -> syn::Result<ToolIo> {
    let ReturnType::Type(_, result) = output else {
        return Err(syn::Error::new(
            output.span(),
            "#[tool] must return Result<T, E>",
        ));
    };
    let types = arguments_of(result, "Result", 2).map_err(|error| {
        syn::Error::new(error.span(), "#[tool] must explicitly return Result<T, E>")
    })?;
    Ok(ToolIo {
        output: types[0].clone(),
        error: types[1].clone(),
    })
}

fn pascal_case(value: &str) -> String {
    let mut result = String::new();
    for part in value.split(|character: char| !character.is_ascii_alphanumeric()) {
        if part.is_empty() {
            continue;
        }
        let mut characters = part.chars();
        if let Some(first) = characters.next() {
            result.extend(first.to_uppercase());
            result.extend(characters);
        }
    }
    if result.is_empty() {
        "Function".into()
    } else {
        result
    }
}

fn expand(mode: Mode, item: Item, root: &Tokens) -> syn::Result<Tokens> {
    match (mode, item) {
        (Mode::Function, Item::Fn(function)) => expand_function(function, root),
        (Mode::Call(method), Item::Impl(implementation)) => {
            expand_impl(implementation, method, false, root)
        }
        (Mode::Signal(method), Item::Impl(implementation)) => {
            expand_impl(implementation, method, true, root)
        }
        (Mode::Function, item) => Err(syn::Error::new(
            item.span(),
            "#[node] requires a free async function; use #[node(call = method)] or #[node(signal = method)] on an inherent impl",
        )),
        (_, item) => Err(syn::Error::new(
            item.span(),
            "call/signal must mark the containing inherent impl, not a standalone method",
        )),
    }
}

struct Io {
    input: Type,
    output: Type,
    error: Type,
}

fn check_signature(signature: &Signature, method: bool, signal: bool) -> syn::Result<Io> {
    if signature.asyncness.is_none() {
        return Err(syn::Error::new(
            signature.span(),
            "node functions and selected methods must be async",
        ));
    }
    if signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
    {
        return Err(syn::Error::new(
            signature.span(),
            "use a regular safe Rust async signature for #[node]",
        ));
    }
    if !signature.generics.params.is_empty() || signature.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            signature.generics.span(),
            "generic node signatures require an explicit Node implementation or a concrete ordinary function",
        ));
    }
    let expected = if method { 3 } else { 2 };
    if signature.inputs.len() != expected {
        return Err(syn::Error::new(
            signature.inputs.span(),
            if method {
                "selected method must take &self, one owned business input, and &Cancellation"
            } else {
                "node function must take one owned business input and Cancellation"
            },
        ));
    }
    let mut arguments = signature.inputs.iter();
    if method {
        let Some(FnArg::Receiver(receiver)) = arguments.next() else {
            return Err(syn::Error::new(
                signature.inputs.span(),
                "selected method must take &self",
            ));
        };
        if receiver.reference.is_none()
            || receiver.mutability.is_some()
            || receiver.colon_token.is_some()
        {
            return Err(syn::Error::new(
                receiver.span(),
                "selected method must take an immutable &self receiver",
            ));
        }
    }
    let Some(FnArg::Typed(input)) = arguments.next() else {
        return Err(syn::Error::new(
            signature.inputs.span(),
            "expected one owned business input",
        ));
    };
    owned(&input.ty)?;
    let Some(FnArg::Typed(cancellation)) = arguments.next() else {
        return Err(syn::Error::new(
            signature.inputs.span(),
            "missing Cancellation parameter",
        ));
    };
    let correct_cancellation = if method {
        matches!(&*cancellation.ty, Type::Reference(reference)
            if reference.mutability.is_none() && named(&reference.elem, "Cancellation"))
    } else {
        named(&cancellation.ty, "Cancellation")
    };
    if !correct_cancellation {
        return Err(syn::Error::new(
            cancellation.ty.span(),
            if method {
                "selected component method must take &Cancellation"
            } else {
                "free node function must take Cancellation by value"
            },
        ));
    }
    let ReturnType::Type(_, result) = &signature.output else {
        return Err(syn::Error::new(
            signature.output.span(),
            "node must return Result<O, E>",
        ));
    };
    let types = arguments_of(result, "Result", 2)?;
    let output = if signal {
        arguments_of(&types[0], "Signal", 1)?[0].clone()
    } else {
        if named(&types[0], "Signal") {
            return Err(syn::Error::new(
                types[0].span(),
                "Signal results require #[node(signal = method)] on an inherent impl",
            ));
        }
        types[0].clone()
    };
    owned(&output)?;
    owned(&types[1])?;
    Ok(Io {
        input: (*input.ty).clone(),
        output,
        error: types[1].clone(),
    })
}

fn named(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none()
        && path.path.segments.last().is_some_and(|segment| segment.ident == name))
}

fn arguments_of(ty: &Type, name: &str, count: usize) -> syn::Result<Vec<Type>> {
    if let Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(segment) = path.path.segments.last()
        && segment.ident == name
        && let PathArguments::AngleBracketed(arguments) = &segment.arguments
    {
        let types = arguments
            .args
            .iter()
            .filter_map(|argument| match argument {
                GenericArgument::Type(ty) => Some(ty.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if types.len() == count && arguments.args.len() == count {
            return Ok(types);
        }
    }
    Err(syn::Error::new(
        ty.span(),
        if name == "Result" {
            "node must explicitly return Result<O, E>"
        } else {
            "signal method must explicitly return Result<node::Signal<O>, E>"
        },
    ))
}

fn owned(ty: &Type) -> syn::Result<()> {
    struct Borrowed(Option<proc_macro2::Span>);
    impl<'ast> Visit<'ast> for Borrowed {
        fn visit_type_reference(&mut self, reference: &'ast syn::TypeReference) {
            self.0.get_or_insert(reference.span());
        }
        fn visit_lifetime(&mut self, lifetime: &'ast syn::Lifetime) {
            self.0.get_or_insert(lifetime.span());
        }
        fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
            if path
                .path
                .segments
                .first()
                .is_some_and(|segment| segment.ident == "Self")
            {
                self.0.get_or_insert(path.span());
            } else {
                visit::visit_type_path(self, path);
            }
        }
    }
    let mut borrowed = Borrowed(None);
    borrowed.visit_type(ty);
    if let Some(span) = borrowed.0 {
        Err(syn::Error::new(
            span,
            "use concrete owned node I/O and error types; borrowed/lifetime/Self-projection signatures require an explicit Node impl",
        ))
    } else {
        Ok(())
    }
}

/// An anonymous, never-executed type-check assertion. Referencing the existing
/// registration method checks precisely its Schema/Serde/error bounds without
/// executing it, constructing a Node, or inventing a node_id or helper API.
fn check(io: &Io, value: Option<&Ident>, ty: Option<&Type>, root: &Tokens) -> Tokens {
    let Io {
        input,
        output,
        error,
    } = io;
    // A user type named N (or any internal-looking name) must not be captured
    // by the assertion's type parameter. Pick a name absent from all I/O tokens.
    let types = quote!(#input #output #error).to_string();
    let index = (0u32..)
        .find(|index| !types.contains(&format!("__HeartNodeDeclaration{index}")))
        .expect("finite type tokens");
    let parameter = Ident::new(
        &format!("__HeartNodeDeclaration{index}"),
        proc_macro2::Span::mixed_site(),
    );
    let value_argument = if value.is_some() {
        quote!(_: #parameter)
    } else {
        quote!()
    };
    let binding = if let Some(value) = value {
        quote!(let callable = #value;)
    } else {
        quote!()
    };
    let assertion = if value.is_some() {
        quote!(let _ = verify(callable);)
    } else {
        let ty = ty.expect("impl type");
        quote!(let _ = verify::<#ty>;)
    };
    quote_spanned! {input.span()=>
        const _: fn() = || {
            #binding
            {
            fn verify<#parameter: #root::agent::node::Node<#input, #output, #error>>(
                #value_argument
            ) {
                let _: fn(
                    &mut #root::agent::workflow::WorkflowBuilder,
                    ::std::string::String, #parameter,
                ) -> ::std::result::Result<
                    &mut #root::agent::workflow::WorkflowBuilder,
                    #root::agent::workflow::GraphError,
                > = #root::agent::workflow::WorkflowBuilder::add_node::<#input, #output, #error, #parameter>;
            }
            #assertion
            }
        };
    }
}

fn expand_function(function: ItemFn, root: &Tokens) -> syn::Result<Tokens> {
    let io = check_signature(&function.sig, false, false)?;
    let assertion = check(&io, Some(&function.sig.ident), None, root);
    Ok(quote!(#function #assertion))
}

fn expand_impl(
    implementation: ItemImpl,
    method: Ident,
    signal: bool,
    root: &Tokens,
) -> syn::Result<Tokens> {
    if implementation.trait_.is_some()
        || implementation.unsafety.is_some()
        || !implementation.generics.params.is_empty()
        || implementation.generics.where_clause.is_some()
    {
        return Err(syn::Error::new(
            implementation.span(),
            "mark a concrete inherent impl; generic or trait impls require an explicit Node impl",
        ));
    }
    let selected = implementation
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == method => Some(function),
            _ => None,
        })
        .ok_or_else(|| {
            syn::Error::new(
                method.span(),
                "selected node method must be defined in this inherent impl",
            )
        })?;
    let io = check_signature(&selected.sig, true, signal)?;
    let Io {
        input,
        output,
        error,
    } = &io;
    let ty = &implementation.self_ty;
    let result = if signal {
        quote!(<#ty>::#method(self, input, &cancellation).await)
    } else {
        quote!(<#ty>::#method(self, input, &cancellation).await.map(#root::agent::node::Signal::Complete))
    };
    let assertion = check(&io, None, Some(ty), root);
    Ok(quote!(
        #implementation
        impl #root::agent::node::Node<#input, #output, #error> for #ty {
            async fn call(&self, input: #input, cancellation: &#root::Cancellation)
                -> ::std::result::Result<#root::agent::node::Signal<#output>, #error>
            { #result }
        }
        #assertion
    ))
}
