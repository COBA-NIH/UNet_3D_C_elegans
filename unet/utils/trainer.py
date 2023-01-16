from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sklearn
import torch
import torch.nn as nn
import shutil

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
        num_epochs=100
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

    def calculate_performance(self, prediction, target, writer, train_or_val):
            # Sigmoid scales values between 0-1
            prediction = prediction.sigmoid().detach().cpu().numpy().flatten()
            target = target.long().detach().cpu().numpy().flatten()

            # Convert probabilities into binary predictions
            th_prediction = (prediction > 0.5)

            # Calculate metrics
            roc = sklearn.metrics.roc_auc_score(target, prediction)
            precision = sklearn.metrics.precision_score(target, th_prediction)
            recall = sklearn.metrics.recall_score(target, th_prediction)
            accuracy = sklearn.metrics.accuracy_score(target, th_prediction)
            f1 = sklearn.metrics.f1_score(target, th_prediction)

            writer.add_scalar(f"{train_or_val}/Accuracy", accuracy, self.epoch+1)
            writer.add_scalar(f"{train_or_val}/Precision", precision, self.epoch+1)
            writer.add_scalar(f"{train_or_val}/Recall", recall, self.epoch+1)
            writer.add_scalar(f"{train_or_val}/F1", f1, self.epoch+1)

            out = f"---- Accuracy: {accuracy} -- Recall: {recall} -- F1: {f1} ----"

            return out

    def fit(self):
        best_val_loss = 0
        for self.epoch in tqdm(range(self.num_epochs)):

            # Train
            self.train()

            # Validate
            current_val_loss = self.validate()

            is_best = current_val_loss > best_val_loss
            best_val_loss = max(current_val_loss, best_val_loss)

            self.scheduler.step(current_val_loss)

            # Save checkpoint
            self.save_checkpoint({
                'epoch': self.epoch + 1,
                'state_dict': self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
            }, is_best)

    def stop_check(self, val_loss, patience=20, delta=0.1):
        return NotImplementedError()
        if self.best_val_score + delta >= val_loss:
            self.counter += 1
            if self.counter >= patience:
                return True


    def save_checkpoint(self, state, is_best, filename="last_checkpoint.pytorch"):  
        torch.save(state, filename)
        if is_best:
            # If the model is best so far, rename it
            shutil.copyfile(filename, "best_checkpoint.pytorch")
        
    def train(self):

        train_loss = 0.0
        train_n = 0

        # Set the model to training mode
        self.model.train()

        for i, (X, y) in enumerate(self.data_loader["train"]):
            # Put the image and the mask on the device
            X, y = X.to(self.device), y.to(self.device)

            # Run a prediction
            prediction = self.model(X)
            # Find the loss of the prediction when compared to the GT
            loss = self.loss_fn(prediction, y)

            # Clear gradients from the optimizer
            self.optimizer.zero_grad()
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

        performance = self.calculate_performance(
            prediction,
            y,
            self.writer,
            "train"
        )
        
        epoch_loss = train_loss / train_n

        self.writer.add_scalar(f"Train-Epoch/Loss", epoch_loss, self.epoch+1)
    
    def validate(self):
        val_loss = 0.0
        val_n = 0
        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(self.data_loader["val"]):
                X, y = X.to(self.device), y.to(self.device)

                prediction = self.model(X)
                loss = self.loss_fn(prediction, y)

                val_loss += loss.item() * X.size(0)
                val_n += X.size(0)

        val_loss = val_loss / val_n

        self.calculate_performance(
                    prediction,
                    y,
                    self.writer,
                    "val"
                    )

        self.writer.add_scalar(f"Val-Epoch/Loss", val_loss, self.epoch+1)

        return val_loss