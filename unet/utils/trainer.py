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

        # For early stopping (patience counter)
        self.counter = 0

        self.writer = SummaryWriter("training_logs")

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
            }, is_best)

            self.stop_check(current_val_loss, best_val_loss)

    def stop_check(self, val_loss, best_val_loss, patience=40, delta=0.1):
        print(f"val_loss: {val_loss}, best_val_loss: {best_val_loss}")
        if best_val_loss + delta > val_loss:
            # Validation accuracy has improved, reset counter
            self.counter = 0
        elif val_loss > best_val_loss + delta:
            # Validation loss has not improved
            self.counter += 1
            if self.counter >= patience:
                print(f"Early stopping at epoch {self.epoch}. Best loss: {best_val_loss}")
                return


    def save_checkpoint(self, 
    state, 
    is_best, 
    filename="last_checkpoint.pytorch"):
        torch.save(state, filename)
        if is_best:
            # If the model is best so far, rename it
            shutil.copyfile(filename, "best_checkpoint.pytorch")

    def update_stats(self, prediction, target, stat_dict):
        # Testing softmax on the class axis 
        prediction = torch.softmax(prediction, axis=1)
        prediction = prediction.detach().cpu().numpy().flatten()
        target = target.long().detach().cpu().numpy().flatten()

        # Convert probabilities into binary predictions
        th_prediction = prediction > 0.5

        # Calculate metrics
        # roc = sklearn.metrics.roc_auc_score(target, prediction)
        precision = sklearn.metrics.precision_score(target, th_prediction)
        recall = sklearn.metrics.recall_score(target, th_prediction)
        accuracy = sklearn.metrics.accuracy_score(target, th_prediction)
        f1 = sklearn.metrics.f1_score(target, th_prediction)
        jaccard = sklearn.metrics.jaccard_score(target, th_prediction)
        dice_coef = self.dice_coef(target, th_prediction)

        stat_dict["precision"] += precision
        stat_dict["recall"] += recall
        stat_dict["accuracy"] += accuracy
        stat_dict["f1"] += f1
        stat_dict["dice_coef"] += dice_coef

        return stat_dict

    def write_epoch_stats(self, stat_dict, train_or_val):
        self.writer.add_scalar(f"{train_or_val}/Accuracy", stat_dict["accuracy"], self.epoch+1)
        self.writer.add_scalar(f"{train_or_val}/Precision", stat_dict["precision"], self.epoch+1)
        self.writer.add_scalar(f"{train_or_val}/Recall", stat_dict["recall"], self.epoch+1)
        self.writer.add_scalar(f"{train_or_val}/F1", stat_dict["f1"], self.epoch+1)
        self.writer.add_scalar(f"{train_or_val}/jaccard", stat_dict["jaccard"], self.epoch+1)
        self.writer.add_scalar(f"{train_or_val}/dice_coef", stat_dict["dice_coef"], self.epoch+1)

    def train(self):
        train_stats = {
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "dice_coef": 0.0
        }
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
            if weight_map is not None:
                loss = self.loss_fn(prediction, y, weight_map)
            else:
                # Since some loss functions are not weighted
                loss = self.loss_fn(prediction, y)

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

            # train_stats = self.update_stats(prediction, y, train_stats)

        # Update stats across all training samples in epoch
        # train_stats = {k: v / train_n for k, v in train_stats.items()}

        # self.write_epoch_stats(train_stats, "train")
        
        epoch_loss = train_loss / train_n

        self.writer.add_scalar(f"Train-Epoch/Loss", epoch_loss, self.epoch+1)

        return epoch_loss, train_stats
    
    def validate(self):
        val_stats = {
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "dice_coef": 0.0
        }
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

                if weight_map is not None:
                    loss = self.loss_fn(prediction, y, weight_map)
                else:
                    loss = self.loss_fn(prediction, y)

                val_loss += loss.item() * X.size(0)
                val_n += X.size(0)

                # val_stats = self.update_stats(prediction, y, val_stats)

        # Update stats across all training samples in epoch
        # val_stats = {k: v / val_n for k, v in val_stats.items()}

        # self.write_epoch_stats(val_stats, "train")

        val_loss = val_loss / val_n

        self.writer.add_scalar(f"Val-Epoch/Loss", val_loss, self.epoch+1)

        return val_loss, val_stats