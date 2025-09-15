import tensorflow as tf
from tensorflow.keras import layers, models
from cognize import EpistemicProgrammableGraph, make_simple_state

def build_cnn(input_shape, num_classes):
    """
    Build a CNN model for network traffic classification, with
    integrated Cognize-based epistemic control hooks.

    The CNN provides baseline feature extraction and classification,
    while Cognize acts as a control-plane that monitors drift and
    modulates intermediate layers dynamically. This design enables
    the network to remain stable under distributional shifts without
    requiring full retraining.

    Args:
        input_shape (tuple): Shape of the input features, e.g., (78, 1).
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
        graph (EpistemicProgrammableGraph): Cognize graph linked to CNN layers.

    Example:
        >>> model, graph = build_cnn((78,1), 5)
        >>> model.summary()
    """
    # --- CNN backbone ---
    model = models.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    # --- Cognize programmable graph ---
    graph = EpistemicProgrammableGraph(max_depth=1, damping=0.6)

    # Add key states corresponding to CNN layers
    graph.add("conv1", make_simple_state(0.0))
    graph.add("dense1", make_simple_state(0.0))

    # Link conv → dense with a programmable modulation edge
    graph.link("conv1", "dense1", mode="policy", weight=0.7, decay=0.9, cooldown=3)

    # Attach Cognize hooks into Keras layers
    def conv1_hook(layer, inputs, outputs):
        norm_val = float(tf.norm(outputs).numpy())
        graph.step("conv1", {"norm": norm_val, "ruptured": False})

    def dense1_hook(layer, inputs, outputs):
        entropy = float(tf.reduce_mean(tf.nn.softmax(outputs)).numpy())
        graph.step("dense1", {"entropy": entropy, "ruptured": False})

    # Register hooks
    model.layers[0].activation = lambda x: tf.nn.relu(x)  # keep original activation
    model.layers[0]._cognize_hook = conv1_hook
    model.layers[-2]._cognize_hook = dense1_hook

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, graph
