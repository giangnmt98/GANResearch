import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import  TensorDataset, ConcatDataset, Subset

import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from ganresearch.utils.utils import create_logger, log_class_distribution


def filter_classes(dataset, desired_classes):
    """
    Lọc dataset chỉ giữ lại các mẫu thuộc các lớp mong muốn.

    Args:
        dataset (torch.utils.data.Dataset): Dataset ban đầu.
        desired_classes (list): Danh sách các lớp mong muốn sử dụng.

    Returns:
        filtered_dataset (Subset): Dataset chỉ chứa các lớp mong muốn.
    """
    indices = [i for i in range(len(dataset)) if dataset[i][1] in desired_classes]
    return Subset(dataset, indices)


def remap_labels(dataset, desired_classes):
    """
    Remap nhãn của các lớp mong muốn để nhãn bắt đầu từ 0.

    Args:
        dataset (torch.utils.data.Dataset): Dataset đã được lọc.
        desired_classes (list): Danh sách các lớp mong muốn.

    Returns:
        remapped_dataset (list): Dataset với nhãn đã được remap.
    """
    class_mapping = {cls: idx for idx, cls in enumerate(desired_classes)}
    remapped_data = [
        (dataset[i][0], class_mapping[dataset[i][1]]) for i in range(len(dataset))
    ]
    return remapped_data


def create_imbalanced_dataset(dataset, imbalance_ratios):
    """
    Tạo một dataset mất cân bằng từ dataset ban đầu.

    Args:
        dataset (torch.utils.data.Dataset): Dataset ban đầu.
        imbalance_ratios (dict): Dictionary chỉ định tỷ lệ mẫu giữ lại cho từng lớp.

    Returns:
        imbalanced_dataset (Subset): Dataset mất cân bằng.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    selected_indices = []

    for cls, ratio in imbalance_ratios.items():
        cls_indices = np.where(labels == cls)[0]
        n_selected = int(len(cls_indices) * ratio)
        selected_indices.extend(
            np.random.choice(cls_indices, n_selected, replace=False)
        )

    return Subset(dataset, selected_indices)


import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import logging

logger = logging.getLogger(__name__)

def load_dataset(
    dataset_name,
    data_dir="./data",
    batch_size=64,
    num_workers=4,
    image_size=(32, 32),
    split_ratios=(0.7, 0.15, 0.15),
    imbalance_ratios=None,
    desired_classes=None,
    apply_imbalance=False,
    logger=None
):
    """
    Load dataset từ các lựa chọn hoặc thư viện PyTorch,
    chỉ sử dụng các lớp mong muốn và tạo mất cân bằng nếu cần.

    Args:
        dataset_name (str): Tên dataset ('MNIST', 'CIFAR10', 'GTRSB', 'ImageFolder').
        data_dir (str): Thư mục chứa dataset.
        batch_size (int): Kích thước batch.
        num_workers (int): Số worker cho DataLoader.
        image_size (tuple): Kích thước ảnh đầu vào (height, width).
        split_ratios (tuple): Tỷ lệ chia train/val/test (tổng là 1.0).
        imbalance_ratios (dict): Tỷ lệ mất cân bằng cho từng lớp (chỉ áp dụng trên train).
        desired_classes (list): Danh sách các lớp mong muốn sử dụng (None nếu muốn sử dụng tất cả).
        apply_imbalance (bool): Có áp dụng mất cân bằng hay không.

    Returns:
        train_loader (DataLoader): DataLoader cho tập huấn luyện (có thể mất cân bằng).
        val_loader (DataLoader): DataLoader cho tập validation.
        test_loader (DataLoader): DataLoader cho tập test (luôn giữ nguyên).
        num_classes (int): Số lớp trong dataset.
        input_channels (int): Số kênh màu của ảnh.
    """
    # Kiểm tra tổng tỷ lệ
    if sum(split_ratios) != 1.0:
        raise ValueError("split_ratios must sum to 1.0")

    # Transform cho ảnh
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize ảnh về kích thước được yêu cầu
            transforms.ToTensor(),         # Chuyển ảnh thành tensor
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # Chuẩn hóa ảnh
        ]
    )

    # Load dataset từ torchvision
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Dữ liệu MNIST có 1 kênh màu
        ])
        full_dataset = datasets.MNIST(
            root=data_dir, train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=True
        )
        num_classes = 10
        input_channels = 1

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(
            root=data_dir, train=True, transform=transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, transform=transform, download=True
        )
        num_classes = 10
        input_channels = 3

    elif dataset_name == "GTRSB":
        full_dataset = datasets.GTSRB(
            root=data_dir, split="train", transform=transform, download=True
        )
        test_dataset = datasets.GTSRB(
            root=data_dir, split="test", transform=transform, download=True
        )
        num_classes = len(
            np.unique([label for _, label in full_dataset])
        )  # Lấy số lớp từ tập train
        input_channels = 3

    elif dataset_name == "ImageFolder":
        full_dataset = ImageFolder(data_dir, transform=transform)
        num_classes = len(full_dataset.classes)
        test_size = int(len(full_dataset) * split_ratios[2])
        train_size = len(full_dataset) - test_size
        full_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        input_channels = 3

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Lọc dataset để chỉ giữ lại các lớp mong muốn
    if desired_classes is not None:
        full_dataset = filter_classes(full_dataset, desired_classes)
        test_dataset = filter_classes(test_dataset, desired_classes)
        full_dataset = remap_labels(full_dataset, desired_classes)
        test_dataset = remap_labels(test_dataset, desired_classes)
        num_classes = len(desired_classes)

    # Chia train/val từ full_dataset
    val_size = int(
        len(full_dataset) * split_ratios[1] / (split_ratios[0] + split_ratios[1])
    )
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Áp dụng imbalance nếu được yêu cầu
    if apply_imbalance and imbalance_ratios:
        train_dataset = create_imbalanced_dataset(train_dataset, imbalance_ratios)

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Hiển thị phân phối số lượng mẫu
    log_class_distribution(train_loader, logger)

    return train_loader, val_loader, test_loader, num_classes, input_channels



def balance_dataset_with_generator(
    generator_file,
    dataloader,
    device,
    noise_dimension,
    save_path=None,
    logger=None,
    gen_has_label=False,
):
    """
    Sử dụng generator để bổ sung dữ liệu cho các class thiểu số và tạo bộ dataset cân bằng.

    Args:
        generator_file (str): Đường dẫn đến file lưu model generator.
        dataloader (DataLoader): DataLoader cung cấp dữ liệu ban đầu (cả ảnh và nhãn).
        device (str): Thiết bị sử dụng (CPU hoặc GPU).
        noise_dimension (int): Kích thước vector noise đầu vào của generator.
        save_path (str, optional): Đường dẫn để lưu hình ảnh được sinh (nếu cần).
        logger (logging.Logger, optional): Logger để ghi log thông tin.
        gen_has_label (bool): Nếu True, generator yêu cầu nhãn đầu vào.

    Returns:
        balanced_dataloader (DataLoader): DataLoader chứa dữ liệu đã được cân bằng.
    """
    # Load generator từ file
    generator = torch.load(generator_file, map_location=device, weights_only=False)
    generator.eval()
    generator.to(device)

    # Tính phân phối số lượng mẫu của từng class
    class_counts = {}
    for _, labels in dataloader:
        labels = labels.numpy()
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    max_samples = max(class_counts.values())
    if logger:
        logger.info(f"Original class distribution: {class_counts}")

    # Tạo dữ liệu bổ sung cho các class thiểu số
    augmented_datasets = []
    for class_idx, count in class_counts.items():
        if count < max_samples:
            additional_samples = int((max_samples - count) / 2)  # Tăng tốc độ tạo dữ liệu
            if logger:
                logger.info(f"Generating {additional_samples} samples for class {class_idx}...")

            # Tạo noise và nhãn
            noise = torch.randn(additional_samples, noise_dimension, 1, 1, device=device)
            labels = torch.full((additional_samples,), class_idx, dtype=torch.long, device=device)

            if gen_has_label:
                # Sinh ảnh từ generator yêu cầu nhãn
                generated_images = generator(noise, labels).detach()
            else:
                # Sinh ảnh từ generator không yêu cầu nhãn
                generated_images = generator(noise).detach()

            # Tạo TensorDataset từ ảnh và nhãn sinh
            augmented_dataset = TensorDataset(generated_images.cpu(), labels.cpu())
            augmented_datasets.append(augmented_dataset)

            # Lưu ảnh (nếu cần)
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                torch.save(generated_images.cpu(), os.path.join(save_path, f"class_{class_idx}_generated.pt"))

    # Kết hợp dữ liệu gốc và dữ liệu bổ sung
    original_dataset = dataloader.dataset
    if augmented_datasets:
        combined_dataset = ConcatDataset([original_dataset, *augmented_datasets])
        if logger:
            logger.info("Dataset balanced successfully.")
    else:
        combined_dataset = original_dataset
        if logger:
            logger.info("Dataset was already balanced.")

    # Tạo hàm custom_collate_fn để ghép batch dữ liệu
    def custom_collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)  # Ghép tensor ảnh
        labels = torch.tensor(labels, dtype=torch.long)  # Chuyển nhãn thành tensor
        return images, labels

    # Tạo DataLoader mới
    balanced_dataloader = DataLoader(
        combined_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        collate_fn=custom_collate_fn,
    )

    return balanced_dataloader

# 1. Định nghĩa model ResNet50Classifier
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        """
        Khởi tạo mô hình ResNet50 với khả năng tùy chỉnh số lớp đầu ra và số kênh đầu vào.

        Args:
            num_classes (int): Số lượng lớp trong dataset.
            input_channels (int): Số kênh đầu vào (1 hoặc 3).
        """
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)

        # Thay đổi tầng conv1 nếu số kênh đầu vào không phải là 3
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=self.backbone.conv1.bias,
            )

        # Thay đổi fully-connected layer cuối cùng
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# 2. Hàm huấn luyện model
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cpu"
):
    """
    Huấn luyện model với train_loader và đánh giá trên val_loader.

    Args:
        model (nn.Module): Mô hình cần huấn luyện.
        train_loader (DataLoader): DataLoader cho tập huấn luyện.
        val_loader (DataLoader): DataLoader cho tập validation.
        criterion (Loss): Hàm mất mát.
        optimizer (Optimizer): Bộ tối ưu.
        num_epochs (int): Số epoch huấn luyện.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
    """
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        train_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )
        for batch_idx, (inputs, labels) in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {running_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}"
        )


# 3. Hàm đánh giá model
def evaluate_model(model, test_loader, generator_file, device='cpu', class_names=None, dataset_name="dataset", is_imbalanced=False, make_balance=False):
    """
    Đánh giá mô hình trên tập test, tính Accuracy, F1-Score tổng thể và từng lớp, tạo biểu đồ và lưu kết quả vào CSV/PNG.

    Args:
        model (nn.Module): Mô hình cần đánh giá.
        test_loader (DataLoader): DataLoader cho tập test.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
        class_names (list, optional): Danh sách tên các lớp (None nếu sử dụng nhãn số).
        dataset_name (str): Tên dataset được sử dụng.
        is_imbalanced (bool): Nếu dataset bị làm mất cân bằng, thư mục sẽ ghi nhận trạng thái này.

    Returns:
        metrics (dict): Bao gồm các chỉ số tổng thể và từng lớp.
    """
    # Xác định thư mục lưu trữ dựa trên tên dataset và trạng thái cân bằng
    balance_status = "imbalanced" if is_imbalanced else "balanced"
    make_balance_status = "make_balance" if make_balance else ""
    if len(make_balance_status) > 0:
        output_dir = os.path.join(f"./results/{dataset_name}_{balance_status}_{make_balance_status}/{generator_file[:-4]}")
    else:
        output_dir = os.path.join(f"./results/{dataset_name}_{balance_status}")
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính chỉ số tổng thể
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Báo cáo chi tiết từng lớp
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Thu thập metric từng lớp
    class_metrics = []
    for class_id, metrics in report.items():
        if class_id.isdigit():  # Chỉ lấy thông tin các lớp
            class_name = class_names[int(class_id)] if class_names else f"Class {class_id}"
            class_metrics.append({
                "class_name": class_name,
                "accuracy": metrics["recall"],  # Accuracy tương ứng với Recall trong báo cáo lớp
                "f1_score": metrics["f1-score"]
            })

    # Thêm giá trị tổng thể (all)
    class_metrics.insert(0, {
        "class_name": "all",
        "accuracy": overall_accuracy,
        "f1_score": overall_f1
    })

    return class_metrics, output_dir

def save_metrics_to_csv(df, output_dir):
    """
    Lưu các chỉ số đánh giá vào file CSV.

    Args:
        df: Danh sách các chỉ số đánh giá từng lớp.
        output_dir (str): Thư mục lưu file CSV.

    Returns:
        str: Đường dẫn đến file CSV đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")

    df.to_csv(csv_path, index=False)

    print(f"Saved evaluation metrics to {csv_path}")


def create_metrics_chart(df, output_dir, dataset_name="dataset", balance_status="balanced", make_balance_status=""):
    """
    Tạo biểu đồ Accuracy và F1-Score cho từng lớp.

    Args:
        class_metrics (list): Danh sách các chỉ số đánh giá từng lớp.
        output_dir (str): Thư mục lưu biểu đồ.
        dataset_name (str): Tên dataset.
        balance_status (str): Trạng thái cân bằng dữ liệu.
        make_balance_status (str): Trạng thái xử lý cân bằng.

    Returns:
        str: Đường dẫn đến file biểu đồ đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # class_names_list = [m["class_name"] for m in class_metrics]
    # accuracies = [m["acc_mean"] for m in class_metrics]
    # f1_scores = [m["f1_mean"] for m in class_metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["class_name"], y=df["acc_mean"], name="Accuracy",
        text=[f"{v:.2f}" for v in df["acc_mean"]], textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=df["class_name"], y=df["f1_mean"], name="F1-Score",
        text=[f"{v:.2f}" for v in df["f1_mean"]], textposition='outside'
    ))

    fig.update_layout(
        title=f"Accuracy and F1-Score for {dataset_name} ({balance_status.capitalize()}{make_balance_status.capitalize()})",
        xaxis_title="Class Names",
        yaxis_title="Metric Values",
        barmode="group",
        legend=dict(title="Metrics")
    )

    chart_path = os.path.join(output_dir, "metrics_chart.html")
    fig.write_html(chart_path)
    print(f"Saved metrics chart to {chart_path}")

def main(dataset_name, data_dir, generator_file, gen_has_label, desired_classes, imbalance_ratios, apply_imbalance=False, make_balance=False, num_epochs=1, lr=0.001):
    logger = create_logger()

    train_loader, val_loader, test_loader, num_classes, input_channels = load_dataset(
        dataset_name=dataset_name,  # Hoặc 'CIFAR10', 'ImageFolder'
        data_dir=data_dir,
        batch_size=64,
        split_ratios=(0.7, 0.1, 0.2),
        imbalance_ratios=imbalance_ratios,
        desired_classes=desired_classes,
        apply_imbalance=apply_imbalance,
        image_size=(64,64),
        logger=logger
    )
    if make_balance==True and apply_imbalance==True:
        # Đường dẫn tới generator đã lưu

        # Cân bằng dataset
        train_loader = balance_dataset_with_generator(
            generator_file=generator_file,
            dataloader=train_loader,
            device="cuda",
            noise_dimension=100,
            save_path="./generated_images",  # Thư mục lưu ảnh sinh
            logger=logger,
            gen_has_label=gen_has_label
        )

        log_class_distribution(train_loader, logger)

    # Khởi tạo model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Classifier(num_classes=num_classes, input_channels=input_channels)

    # Cấu hình optimizer và loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Evaluate model
    logger.info("Evaluating model on test data:")
    results, output_dir = evaluate_model(model, test_loader, generator_file=generator_file, dataset_name=dataset_name, device=device, is_imbalanced=apply_imbalance, make_balance=make_balance)
    return results, output_dir


def process_metrics(df_results, save_dir):
    results = df_results.groupby("class_name").agg(
        acc_mean=("accuracy", "mean"),
        acc_se=("accuracy", sem),  # Standard error of accuracy
        f1_mean=("f1_score", "mean"),
        f1_se=("f1_score", sem)  # Standard error of f1
    ).reset_index()

    save_metrics_to_csv(results, save_dir)
    return results

def run_main_for_iterations(num_iterations, dataset, data_directory, apply_imbalance, balance, desired_classes, imbalance_ratios, generator_file, gen_has_label):
    output_dir = None
    results_list = []

    for _ in range(num_iterations):
        result, output_dir = main(
            dataset_name=dataset,
            data_dir=data_directory,
            desired_classes=desired_classes,
            imbalance_ratios=imbalance_ratios,
            apply_imbalance=apply_imbalance,
            make_balance=balance,
            num_epochs=10,
            generator_file=generator_file,
            gen_has_label=gen_has_label
        )
        tmp_df = pd.DataFrame(result)
        results_list.append(tmp_df)

    combined_results = pd.concat(results_list)
    return combined_results, output_dir

# 4. Main pipeline
if __name__ == "__main__":
    # Load dataset
    desired_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Lọc lớp mong muốn

    # imbalance_ratios = {
    #     0: 1,
    #     1: 1,
    #     2: 1,
    #     3: 0.5,
    #     4: 0.4,
    #     5: 0.3,
    #     6: 0.3,
    #     7: 0.1,
    #     8: 0.1,
    #     9: 0.1,
    # }  # Tạo mất cân bằng cho custom data

    from scipy.stats import sem  # Import hàm tính sai số chuẩn
    import pandas as pd

    # Parameters
    # DATASET_NAME = "ImageFolder"
    # DATA_DIR = "data/Stanford Dogs Dataset"
    DATASET_NAME = "ImageFolder"
    DATA_DIR = "data/Stanford Dogs Dataset"
    APPLY_IMBALANCE = True
    MAKE_BALANCE = True
    DESIRED_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # This should be defined
    IMBALANCE_RATIOS = {
           0: 1,
           1: 1,
           2: 1,
           3: 0.5,
           4: 0.4,
           5: 0.3,
           6: 0.3,
           7: 0.1,
           8: 0.1,
           9: 0.1,
    }  # This should be defined

    # Run main
    df_results, output_dir = run_main_for_iterations(5, DATASET_NAME, DATA_DIR, APPLY_IMBALANCE, MAKE_BALANCE,
                                                     DESIRED_CLASSES, IMBALANCE_RATIOS, generator_file="experiments/cgan/standford_dogs/balance/non_lc/20241127_173522/generator_final.pth", gen_has_label=True)
    # df_results, output_dir = run_main_for_iterations(5, DATASET_NAME, DATA_DIR, APPLY_IMBALANCE, MAKE_BALANCE,
    #                                                  DESIRED_CLASSES, IMBALANCE_RATIOS,
    #                                                  generator_file=None,
    #                                                  gen_has_label=False)
    # Process metrics
    processed_results = process_metrics(df_results, output_dir)

    # Create metrics chart
    balance_status = "imbalanced" if APPLY_IMBALANCE else "balanced"
    create_metrics_chart(processed_results, output_dir, dataset_name=DATASET_NAME, balance_status=balance_status)


    # main(dataset_name="ImageFolder", data_dir= "data/Stanford Dogs Dataset", desired_classes=desired_classes,
    #      imbalance_ratios=imbalance_ratios, apply_imbalance=True, make_balance=True, num_epochs=10,
    #      generator_file="experiments/dcgan/standford_dogs/balance/lc_0.3/20241127_161530/generator_final.pth", gen_has_label=False)