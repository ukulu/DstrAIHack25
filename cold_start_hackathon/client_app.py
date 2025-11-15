import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from cold_start_hackathon.task import Net, load_data
from cold_start_hackathon.util import PARTITION_HOSPITAL_MAP
from cold_start_hackathon.task import test as test_fn
from cold_start_hackathon.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    
    my_key = "elias"
    if my_key in context.state.keys():
        context.state[my_key]["round"] += 1
    else:
        context.state[my_key] = ConfigRecord({"round": 0})

    # Load the model and initialize it with the received weights
    model = Net(context.run_config["model"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    trainloader = load_data(dataset_name, "train", image_size=image_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        context.run_config["max-batches-per-round"],
        context.state[my_key]["round"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "partition-id": context.node_config["partition-id"],
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net(context.run_config["model"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    valloader = load_data(dataset_name, "eval", image_size=image_size)

    eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)

    metric_record = MetricRecord({
        "partition-id": context.node_config["partition-id"],
        "eval_loss": eval_loss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num-examples": len(valloader.dataset),
        "probs": probs.tolist(),  # Convert numpy array to list for MetricRecord
        "labels": labels.tolist(),  # Convert numpy array to list for MetricRecord
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
