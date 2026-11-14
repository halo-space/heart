use super::*;
use halo::agent::evaluation::{Evaluation, Evaluator};
use halo::agent::plan::{Edge, Plan, PlanNode};
use futures_executor::block_on;

struct Accept;

impl Evaluator for Accept {
    async fn evaluate(
        &self,
        _input: &Input,
        _result: &Messages,
        _cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        Ok(Evaluation {
            passed: true,
            feedback: None,
        })
    }
}

#[test]
fn evaluator_returns_final_result_judgment_without_node_identity() {
    let evaluation = block_on(Accept.evaluate(
        &Input::default(),
        &Messages::default(),
        &Cancellation::new(),
    ))
    .unwrap();
    assert!(evaluation.passed);
    assert!(evaluation.feedback.is_none());
}

fn node(id: &str, name: &str) -> PlanNode {
    PlanNode {
        node_id: id.into(),
        name: name.into(),
        objective: format!("run {name}"),
    }
}

#[test]
fn plan_operations_are_atomic_and_preserve_dag_rules() {
    let mut plan = Plan::new(1).unwrap();
    plan.add(node("a", "search"), vec![], vec![]).unwrap();
    plan.add(node("b", "summarize"), vec!["a".into()], vec![])
        .unwrap();
    let before = plan.clone();
    plan.add(node("c", "any task"), vec!["b".into()], vec![])
        .unwrap();
    assert_ne!(plan, before);

    plan.insert("a", node("d", "review"), "b").unwrap();
    assert!(plan.edges.contains(&Edge {
        from: "a".into(),
        to: "d".into()
    }));
    assert!(plan.edges.contains(&Edge {
        from: "d".into(),
        to: "b".into()
    }));
}

#[test]
fn delete_removes_the_node_and_all_successors() {
    let mut plan = Plan::new(1).unwrap();
    plan.add(node("a", "search"), vec![], vec![]).unwrap();
    plan.add(node("b", "summarize"), vec!["a".into()], vec![])
        .unwrap();
    plan.add(node("c", "review"), vec!["b".into()], vec![])
        .unwrap();
    plan.delete("b").unwrap();
    assert_eq!(plan.nodes.keys().collect::<Vec<_>>(), vec![&"a".to_owned()]);
    assert!(plan.edges.is_empty());
}

#[test]
fn validation_rejects_cycle_and_unknown_edges() {
    let mut plan = Plan::new(1).unwrap();
    plan.add(node("a", "search"), vec![], vec![]).unwrap();
    plan.add(node("b", "summarize"), vec!["a".into()], vec![])
        .unwrap();
    let before = plan.clone();
    assert!(
        plan.add(node("c", "review"), vec!["missing".into()], vec![])
            .is_err()
    );
    assert_eq!(plan, before);
    plan.edges.push(Edge {
        from: "b".into(),
        to: "a".into(),
    });
    assert!(plan.validate().is_err());
}

#[test]
fn plan_json_uses_node_map_keys_as_logical_ids() {
    let mut plan = Plan::new(2).unwrap();
    plan.add(node("search_a", "search"), vec![], vec![])
        .unwrap();
    let value = serde_json::to_value(&plan).unwrap();
    assert_eq!(value["version"], 2);
    assert_eq!(value["nodes"]["search_a"]["name"], "search");
    assert!(value["nodes"]["search_a"].get("node_id").is_none());
    let round_trip: Plan = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip.nodes["search_a"].node_id, "search_a");
}

fn application_id() -> i64 {
    panic!("the application ID function must not run during assembly")
}

fn uncalled_provider_chat() -> impl Chat {
    halo::agent::core::provider::vllm::Vllm::builder()
        .api_key("compile-only")
        .build()
        .expect("construct provider without sending a request")
        .chat(halo::agent::model::chat::Config::new("compile-only"))
        .expect("construct Provider Chat")
}

#[test]
fn id_function_is_optional_and_is_not_called_by_build() {
    let agent = halo::agent::react()
        .model(uncalled_provider_chat())
        .build()
        .expect("react agent without an ID source builds");
    assert!(agent.id_fn.is_none());

    let configured = halo::agent::react()
        .with_id_fn(application_id)
        .model(uncalled_provider_chat())
        .build()
        .expect("react agent with an application ID source builds");
    assert_eq!(
        configured.id_fn.map(|id| id as usize),
        Some(application_id as *const () as usize)
    );
}

#[test]
fn id_function_survives_model_rebinding_without_memory() {
    let configured = halo::agent::react()
        .model(uncalled_provider_chat())
        .with_id_fn(application_id)
        .model(uncalled_provider_chat())
        .build()
        .expect("model rebinding keeps ID configuration");
    assert_eq!(
        configured.id_fn.map(|id| id as usize),
        Some(application_id as *const () as usize)
    );
}
