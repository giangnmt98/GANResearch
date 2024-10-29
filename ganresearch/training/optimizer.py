import torch.optim as optim


class Optimizer:
    def __init__(self, config):
        """
        Khởi tạo Optimizer với config.

        Args:
            config (dict): Cấu hình chứa các tham số cho optimizer.
        """
        self.config = config

    def create(self, model_params):
        """
        Tạo optimizer dựa trên thuật toán được chỉ định trong config.

        Args:
            model_params (iterable): Tham số của mô hình (generator hoặc discriminator).

        Returns:
            torch.optim.Optimizer: Optimizer cho mô hình.
        """
        optimizer_type = self.config["training"].get("optimizer", "adam").lower()

        if optimizer_type == "adam":
            return optim.Adam(
                model_params,
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], 0.999),
            )
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(
                model_params, lr=self.config["training"]["learning_rate"]
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                model_params, lr=self.config["training"]["learning_rate"], momentum=0.9
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model_params,
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], 0.999),
            )
        else:
            raise ValueError(f"Optimizer `{optimizer_type}` không được hỗ trợ.")
