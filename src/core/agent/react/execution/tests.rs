use super::*;

#[test]
fn planned_task_instruction_is_merged_into_the_existing_system_message() {
    let mut system = Message::new(Role::System);
    system.content.push(Part {
        r#type: "text".into(),
        data: part_data(json!({"value": "parent instructions"})),
    });
    let user = Message::new(Role::User);
    let input = Message::new(Role::User);

    let messages = task_messages(Messages::new([system, user]), input);

    assert_eq!(
        messages
            .0
            .iter()
            .map(|message| message.role.clone())
            .collect::<Vec<_>>(),
        [Role::System, Role::User, Role::User]
    );
    assert_eq!(messages.0[0].content.len(), 2);
    assert_eq!(
        messages.0[0].content[0].data["value"],
        "parent instructions"
    );
    assert!(
        messages.0[0].content[1].data["value"]
            .as_str()
            .unwrap()
            .contains("current planned task objective")
    );
}
