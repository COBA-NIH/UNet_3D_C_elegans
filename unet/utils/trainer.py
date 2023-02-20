from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sklearn
import torch
import torch.nn as nn
import shutil
import numpy as np
import unet.utils.metrics as metrics

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
        neptune_run=None
        ):
        self.model = model
        self.device = device
        assert isinstance(data_loader, dict)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.neptune_run = neptune_run

        # For early stopping (patience counter)
        self.counter = 0

        self.writer = SummaryWriter("training_logs")

    def fit(self):
        best_val_loss = np.inf
        for epoch in tqdm(range(self.num_epochs)):
            self.epoch = epoch

            # Train
            _ = self.train()

            # Validate
            current_val_loss = self.validate()

            # if self.stop_check(current_val_loss, best_val_loss):
            #     # If True, stop fit
            #     break

            is_best = current_val_loss < best_val_loss
            print(is_best, best_val_loss, current_val_loss)
            best_val_loss = min(current_val_loss, best_val_loss)

            self.scheduler.step(current_val_loss)

            # Save checkpoint
            self.save_checkpoint({
                "epoch": self.epoch + 1,
                "state_dict": self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict(),
            }, is_best)


    def stop_check(self, val_loss, best_val_loss, patience=40, delta=0):
        print(f"val_loss: {val_loss}, best_val_loss: {best_val_loss}")
        if best_val_loss > (val_loss + delta):
            # Validation accuracy has improved, reset counter
            self.counter = 0
        elif (val_loss + delta) >= best_val_loss:
            # Validation loss has not improved
            self.counter += 1
            print(f"Counter: {self.counter}/{self.patience}")
            if self.counter >= patience:
                print(f"Early stopping at epoch {self.epoch}. Best loss: {best_val_loss}")
                return True
        elif self.optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate has reached the minimum")
            return True
        else:
            return False

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
        dice_coef = metrics.dice_coef(target, th_prediction)

        stat_dict["precision"] += precision
        stat_dict["recall"] += recall
        stat_dict["accuracy"] += accuracy
        stat_dict["f1"] += f1
        stat_dict["dice_coef"] += dice_coef

        return stat_dict


    def train(self):
        train_loss = 0.0
        train_n = 0
        dice_coef = 0.0
        rand_error = 0.0

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

            # Gather the loss
            # X.size(0) is just the batch size
            # Here, we multiply with the constant batch size so that variations in batch size
            # are accounted for (eg last mini-batch is smaller)
            train_loss += loss.item() * X.size(0)
            # Increment the total number of training samples based on the batch size
            # Used for calculating average metrics later
            train_n += X.size(0)

            # Calculate the dice coefficient for recording
            dice_coef += metrics.dice_coef(prediction.sigmoid(), y) * X.size(0)
            # calculate_rand_error returns an array, with each element the rand error at a given threshold
            # The default is 0.4
            rand_error += metrics.calculate_rand_error(prediction.sigmoid(), y)[0] * X.size(0)

            # Update the the parameters of the model
            self.optimizer.step()


        
        epoch_train_loss = train_loss / train_n
        epoch_dice_coef = dice_coef / train_n
        epoch_rand_error = rand_error / train_n

        if self.neptune_run is not None:
            self.neptune_run["train/loss"].append(epoch_train_loss)
            self.neptune_run["train/dice_coef"].append(epoch_dice_coef)
            self.neptune_run["train/rand_error"].append(epoch_rand_error)

        self.writer.add_scalar(f"train/loss", epoch_train_loss, self.epoch+1)

        return epoch_train_loss
    
    def validate(self):
        val_loss = 0.0
        val_n = 0
        dice_coef = 0.0
        rand_error = 0.0
        self.model.eval()
        with torch.no_grad():
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

                dice_coef += metrics.dice_coef(prediction.sigmoid(), y) * X.size(0)
                rand_error += metrics.calculate_rand_error(prediction.sigmoid(), y)[0] * X.size(0)

        epoch_val_loss = val_loss / val_n
        epoch_dice_coef = dice_coef / val_n
        epoch_rand_error = rand_error / val_n

        if self.neptune_run is not None:
            self.neptune_run["val/loss"].append(epoch_val_loss)
            self.neptune_run["val/dice_coef"].append(epoch_dice_coef)
            self.neptune_run["val/rand_error"].append(epoch_rand_error)

        self.writer.add_scalar(f"val/loss", epoch_val_loss, self.epoch+1)

        return epoch_val_loss