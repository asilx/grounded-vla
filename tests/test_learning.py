import pytest

torch = pytest.importorskip("torch")

from grounded_vla.learning.adapter import GatedKnowledgeAdapter, GraphContextEncoder  # noqa: E402
from grounded_vla.learning.losses import flow_matching_loss, masked_predicate_loss  # noqa: E402


def graph():
    torch.manual_seed(13)
    features = torch.randn(2, 4, 6)
    edges = torch.randint(0, 3, (2, 4, 4))
    valid = torch.ones(2, 4, dtype=torch.bool)
    return features, edges, valid


def test_encoder_is_invariant_to_consistent_object_permutation():
    features, edges, valid = graph()
    encoder = GraphContextEncoder(6, 16, 4, 3, 3).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    a, _ = encoder(features, edges, valid)
    b, _ = encoder(
        features[:, permutation], edges[:, permutation][:, :, permutation], valid[:, permutation]
    )
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_all_unknown_graph_is_exact_identity_even_with_trained_biases():
    features, edges, valid = graph()
    model = GatedKnowledgeAdapter(6, 16, 4, 3, 3, initial_gate=1.0)
    hidden = torch.randn(2, 5, 16)
    valid[:] = False
    features[:] = float("nan")
    with torch.no_grad():
        model.attention.out_proj.bias.fill_(3.0)
    actual = model(hidden, features, edges, valid)
    assert torch.equal(actual, hidden)


def test_empty_graph_is_identity():
    model = GatedKnowledgeAdapter(6, 16, 4, 3, 3)
    hidden = torch.randn(2, 5, 16)
    actual = model(
        hidden,
        torch.empty(2, 0, 6),
        torch.empty(2, 0, 0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.bool),
    )
    assert torch.equal(actual, hidden)


def test_gradients_reach_adapter_through_frozen_downstream_layer():
    features, edges, valid = graph()
    adapter = GatedKnowledgeAdapter(6, 16, 4, 3, 3)
    frozen = torch.nn.Linear(16, 4).requires_grad_(False)
    hidden = torch.randn(2, 5, 16)
    frozen(adapter(hidden, features, edges, valid)).square().mean().backward()
    assert adapter.encoder.input.weight.grad.abs().sum() > 0
    assert adapter.encoder.relation_bias.weight.grad.abs().sum() > 0
    assert adapter.gate.grad.abs() > 0
    assert all(parameter.grad is None for parameter in frozen.parameters())


def test_unknown_labels_have_zero_gradient_and_do_not_mean_false():
    logits = torch.tensor([0.5, -0.8, 2.0], requires_grad=True)
    labels = torch.tensor([1, 0, -1])
    masked_predicate_loss(logits, labels).backward()
    assert logits.grad[2] == 0
    assert logits.grad[0] < 0 and logits.grad[1] > 0


def test_flow_target_sign_and_episode_padding():
    actions = torch.ones(1, 2, 3)
    noise = torch.zeros_like(actions)
    velocity = torch.full_like(actions, -1.0).requires_grad_()
    valid = torch.tensor([[True, False]])
    actions[:, 1] = float("nan")
    loss = flow_matching_loss(velocity, actions, noise, valid)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(velocity.grad).all()
    assert torch.equal(velocity.grad, torch.zeros_like(velocity))


def test_completely_unknown_loss_is_finite_zero():
    logits = torch.full((3,), float("nan"), requires_grad=True)
    loss = masked_predicate_loss(logits, torch.full((3,), -1))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
