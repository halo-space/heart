use halo::agent::Error;
use halo::agent::node::Echo;
use halo::agent::workflow::{
    Builder,
    graph::{END, START},
};

#[test]
fn builder_rejects_a_cycle_outside_an_explicit_loop() {
    let mut builder = Builder::new("direct-cycle");
    builder.add_node("a", Echo).unwrap();
    builder.add_node("b", Echo).unwrap();
    builder.add_edge(START, "a").unwrap();
    builder.add_edge("a", "b").unwrap();
    builder.add_edge("b", "a").unwrap();
    builder.add_edge("b", END).unwrap();

    let error = builder.compile().unwrap_err();
    assert_eq!(error.code, "GRAPH_CYCLE");
}

#[test]
fn builder_allows_one_condition_and_one_explicit_loop_per_source() {
    let mut builder = Builder::new("relation-duplicates");
    for id in ["source", "left", "right"] {
        builder.add_node(id, Echo).unwrap();
    }
    builder
        .add_conditional_edges(
            "source",
            |_messages| Ok::<_, Error>(vec!["left".into()]),
            ["left"],
        )
        .unwrap();
    let error = match builder.add_conditional_edges(
        "source",
        |_messages| Ok::<_, Error>(vec!["right".into()]),
        ["right"],
    ) {
        Ok(_) => panic!("duplicate condition was accepted"),
        Err(error) => error,
    };
    assert_eq!(error.code, "DUPLICATE_CONDITION");

    let mut loops = Builder::new("loop-duplicates");
    for id in ["source", "target"] {
        loops.add_node(id, Echo).unwrap();
    }
    loops.add_loop_edge("source", "target", |_| true).unwrap();
    let error = match loops.add_loop_edge("source", "target", |_| true) {
        Ok(_) => panic!("duplicate loop was accepted"),
        Err(error) => error,
    };
    assert_eq!(error.code, "DUPLICATE_LOOP");
}
