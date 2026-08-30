def train_and_log(alpha = 0.0001, learning_rate = 0.001, batch_size = 64, run_name=None):
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("Alpha", alpha)
        mlflow.log_param("Learning Rate", learning_rate)
        mlflow.log_param("Batch Size", batch_size)

        model, train_acc, acc, f1 = train_and_evaluate(alpha = alpha, learning_rate_init=learning_rate, batch_size=batch_size)

        # --- metrics (at least 2) ---
        mlflow.log_metric("training accuracy", train_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        mlflow.set_tag("Assignment 1",'Test Run')
        mlflow.sklearn.log_model(model,
        name="model",
        serialization_format="skops",
        skops_trusted_types=[
            "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
        ])

        run_id = mlflow.active_run().info.run_id
        print(f"Logged run {run_id}  |  acc={acc:.4f}  f1={f1:.4f} train_acc={train_acc:.4f}")
        return run_id