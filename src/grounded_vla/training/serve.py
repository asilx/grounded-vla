"""Serve the trained π0 graph adapter through the official openpi websocket protocol."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--disable-graph", action="store_true")
    args = parser.parse_args()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    from grounded_vla.training.inference import TrainedPi0Policy

    policy = TrainedPi0Policy(
        args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        num_steps=args.num_steps,
        disable_graph=args.disable_graph,
    )
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
