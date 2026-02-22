"""
Compatibility wrapper for federated learning client.
Keeps legacy imports working while delegating to the maintained implementation.
"""

from typing import Optional

from client.fl_client import FlowerClient, create_flower_client


def create_client(
    client_id: int,
    data_path: str,
    local_epochs: int = 5,
    learning_rate: float = 0.001,
    apply_smote: bool = True
) -> FlowerClient:
    """
    Legacy factory function retained for backward compatibility.

    Args:
        client_id: Unique identifier for the client
        data_path: Path to the client's local dataset
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for local training
        apply_smote: Retained for compatibility; SMOTE is not applied here

    Returns:
        Initialized FlowerClient instance
    """
    return create_flower_client(
        client_id=client_id,
        client_data_path=data_path,
        num_features=None,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )


__all__ = ["FlowerClient", "create_client"]

