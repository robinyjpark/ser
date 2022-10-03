import torch


def infer(images, model):
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = float(max(list(torch.exp(output)[0])))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"The predicted label is: {pred}")
    print(f"The certainty is: {certainty}")


def load_model(run_path, label, dataloader):
    # select image to run inference for
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")
    return images, model


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
