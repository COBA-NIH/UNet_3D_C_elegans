from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sklearn
import torch
import torch.nn as nn
import shutil
import numpy as np

class RunTraining:
    """This class runs training and validation"""
    def __init__(
        self, 
        model, 
        device, 
        data_loader, 
        loss_fn, 
        optimizer, 
        scheduler,
        num_epochs=100,
        ):
        self.model = model
        self.device = device
        assert isinstance(data_loader, dict)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        # For early stopping
        self.best_val_score = 0
        self.counter = 0

        self.writer = SummaryWriter("training_logs")

    def calculate_performance(self, prediction, target, iterable, train_or_val):
        """Assumes output from model is sigmoidal"""
        # prediction = prediction.detach().sigmoid().cpu().numpy().flatten()

        # Testing softmax on the class axis 
        prediction = torch.softmax(prediction, axis=1)
        prediction = prediction.detach().cpu().numpy().flatten()
        target = target.long().detach().cpu().numpy().flatten()

        # Convert probabilities into binary predictions
        th_prediction = prediction > 0.5

        # Dict to hold stats
        stats = {}

        # Calculate metrics
        # roc = sklearn.metrics.roc_auc_score(target, prediction)
        stats["precision"] = sklearn.metrics.precision_score(target, th_prediction)
        stats["recall"] = sklearn.metrics.recall_score(target, th_prediction)
        stats["accuracy"] = sklearn.metrics.accuracy_score(target, th_prediction)
        stats["f1"] = sklearn.metrics.f1_score(target, th_prediction)
        stats["jaccard"] = sklearn.metrics.jaccard_score(target, th_prediction)
        stats["dice_coef"] = self.dice_coef(target, th_prediction)

        self.writer.add_scalar(f"{train_or_val}/Accuracy", stats["accuracy"], iterable+1)
        self.writer.add_scalar(f"{train_or_val}/Precision", stats["precision"], iterable+1)
        self.writer.add_scalar(f"{train_or_val}/Recall", stats["recall"], iterable+1)
        self.writer.add_scalar(f"{train_or_val}/F1", stats["f1"], iterable+1)
        self.writer.add_scalar(f"{train_or_val}/jaccard", stats["jaccard"], iterable+1)
        self.writer.add_scalar(f"{train_or_val}/dice_coef", stats["dice_coef"], iterable+1)

        return stats

    def dice_coef(self, y_true, y_pred):
        intersection = np.sum((y_true+y_pred == 2))
        union = np.sum(y_true) + np.sum(y_pred)
        dice = (2 * intersection) / (union + 1e-6)
        return dice

    def fit(self):
        best_val_loss = np.inf
        for epoch in tqdm(range(self.num_epochs)):
            self.epoch = epoch

            # Train
            _, _ = self.train()

            # Validate
            current_val_loss, _ = self.validate()

            is_best = current_val_loss < best_val_loss
            print(is_best, best_val_loss, current_val_loss)
            best_val_loss = min(current_val_loss, best_val_loss)

            self.scheduler.step(current_val_loss)

            # Save checkpoint
            self.save_checkpoint({
                "epoch": self.epoch + 1,
                "state_dict": self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict(),
                # "optimizer" : self.optimizer.state_dict(),
                # "scheduler" : self.scheduler.state_dict(),
            }, is_best)

    # def stop_check(self, val_loss, patience=20, delta=0.1):
    #     return NotImplementedError()
    #     if self.best_val_score + delta >= val_loss:
    #         self.counter += 1
    #         if self.counter >= patience:
    #             return True


    def save_checkpoint(self, 
    state, 
    is_best, 
    filename="last_checkpoint.pytorch"):
        torch.save(state, filename)
        if is_best:
            # If the model is best so far, rename it
            shutil.copyfile(filename, "best_checkpoint.pytorch")
        
    def train(self):

        train_loss = 0.0
        train_n = 0

        # Set the model to training mode
        self.model.train()

        for i, data in enumerate(self.data_loader["train"]):
            X, y = data["image"].to(self.device), data["mask"].to(self.device)
            
            # Check if there is a weight map
            try:
                weight_map = data["weight_map"].to(self.device)
            except:
                weight_map = None
            
            # Clear gradients from the optimizer
            self.optimizer.zero_grad()

            # Run a prediction
            prediction = self.model(X)

            # Find the loss of the prediction when compared to the GT
            loss = self.loss_fn(prediction, y, weight_map)

            # Using this loss, calculate the gradients of loss for all parameters
            loss.backward()
            
            # Update the the parameters of the model
            self.optimizer.step()

            # Gather the loss
            # X.size(0) is just the batch size
            # Here, we multiply with the constant batch size so that variations in batch size
            # are accounted for (eg last mini-batch is smaller)
            train_loss += loss.item() * X.size(0)
            # Increment the total number of training samples based on the batch size
            # Used for calculating average metrics later
            train_n += X.size(0)

        stats = self.calculate_performance(
            prediction,
            y,
            self.epoch,
            "train"
        )
        
        epoch_loss = train_loss / train_n

        self.writer.add_scalar(f"Train-Epoch/Loss", epoch_loss, self.epoch+1)

        return epoch_loss, stats
    
    def validate(self):
        val_loss = 0.0
        val_n = 0
        self.model.eval()
        with torch.no_grad():
            # for i, (X, y, weight_map) in enumerate(self.data_loader["val"]):
                # X, y, weight_map = X.to(self.device), y.to(self.device), weight_map.to(self.device)
            for i, data in enumerate(self.data_loader["val"]):
                X, y = data["image"].to(self.device), data["mask"].to(self.device)
                
                # Check if there is a weight map
                try:
                    weight_map = data["weight_map"].to(self.device)
                except:
                    weight_map = None

                prediction = self.model(X)
                loss = self.loss_fn(prediction, y, weight_map)

                val_loss += loss.item() * X.size(0)
                val_n += X.size(0)

        val_loss = val_loss / val_n

        stats = self.calculate_performance(
                    prediction,
                    y,
                    self.epoch,
                    "val"
                    )

        self.writer.add_scalar(f"Val-Epoch/Loss", val_loss, self.epoch+1)

        return val_loss, stats