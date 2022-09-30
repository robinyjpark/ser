import torch
import torch.nn.functional as F
import json


def model_train(
    epochs,
    training_dataloader,
    validation_dataloader,
    device,
    model,
    optimizer,
    output_path,
):

    prev_val_acc = 0
    for epoch in range(epochs):
        train(epoch, device, training_dataloader, model, optimizer)
        val_acc = validate(epoch, device, validation_dataloader, model)
        if val_acc > prev_val_acc:
            # save the model with the best accuracy
            torch.save(model.state_dict(), f"{output_path}/model.pt")
            prev_val_acc = val_acc

            # save the associated epoch and accuracy
            perform_dict = {
                "epoch": epoch,
                "val_acc": val_acc,
            }

            param_path = f"{output_path}/best_results.json"
            with open(param_path, "w") as outfile:
                json.dump(perform_dict, outfile)


def train(epoch, device, training_dataloader, model, optimizer):
    for i, (images, labels) in enumerate(training_dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )


def validate(epoch, device, validation_dataloader, model):
    # validate
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(validation_dataloader.dataset)
        val_acc = correct / len(validation_dataloader.dataset)
        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")
    return val_acc
