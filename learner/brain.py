import os
import shutil
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


class TrainingManager:

    def __init__(
        self,
        data,
        model,
        result_dir,
        config,
        logger,
    ):
        self.data = data
        self.model = model
        self.result_dir = result_dir
        self.config = config
        self.logger = logger

        self.optimizer_instance = None
        self.scheduler_instance = None

        self.__init_training_manager()

    def __init_training_manager(self):
        self.encounter_nan = False
        self.best_model = None

        self.__initialize_data()
        self.__initialize_model()
        self.__initialize_optimizer()
        self.__initialize_scheduler()
        self.__initialize_results()

    def __initialize_data(self):
        train_loader, val_loader = self.data
        self.train_loader = train_loader
        self.val_loader = val_loader

    def __initialize_model(self):
        self.model = self.model.to(self.config.device)
        self.model = self.model.to(self.config.dtype)

    def __initialize_optimizer(self):
        optimizer = self.config.optimizer
        learning_rate = self.config.learning_rate

        # Switch optimizer from Adam to LBFGS
        if optimizer == "adam_LBFGS":
            optimizer = "adam"
        elif optimizer == "adam_LBFGS_next":
            optimizer = "LBFGS"
            learning_rate = 1.0

        # train some parameters
        # base_param_ids = set(map(id, self.net.forcesNet.parameters()))
        # new_params = [p for p in self.net.parameters() if id(p) not in base_param_ids]
        # param_groups = [{'params': self.net.forcesNet.parameters(), 'lr': self.lr * 10},
        #                 {'params': new_params, 'lr': self.lr}]
        # self.__optimizer = torch.optim.Adam(param_groups, lr=self.lr,
        #                                     weight_decay=1e-4,
        #                                     betas=(0.9, 0.999))

        if optimizer == "adam":
            self.optimizer_instance = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "LBFGS":
            self.optimizer_instance = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

    def __initialize_scheduler(self):
        scheduler = self.config.scheduler
        if scheduler == "no_scheduler":
            self.scheduler_instance = None
        elif scheduler == "StepLR":
            self.scheduler_instance = torch.optim.lr_scheduler.StepLR(self.optimizer_instance, step_size=int(self.iterations // 5), gamma=0.7)

    def __initialize_results(self):
        """
        Initialize result directories and data structures.
        """
        # Create a directory to store model training files
        output_dir = self.result_dir
        model_output_dir = os.path.join(output_dir, "training_file")
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
        os.makedirs(model_output_dir)
        self.model_output_dir = model_output_dir

        # Create a directory to store inference results
        evaluate_output_dir = os.path.join(output_dir, "infer")
        if os.path.exists(evaluate_output_dir):
            shutil.rmtree(evaluate_output_dir)
        os.makedirs(evaluate_output_dir)
        self.evaluate_output_dir = evaluate_output_dir

        self.results = {}

    def run_training(self):
        self.logger.info("Training...")
        loss_history = []
        pbar = tqdm(range(self.config.iterations + 1), desc="Processing")
        for iter in pbar:
            # Train ---------------------------------------------------------------
            for batch_data in self.train_loader:
                loss_temp_list = []
                self.model.train()

                def closure():
                    loss = self.model.criterion(data=batch_data,current_iterations=iter)
                    loss_temp_list.append(loss)
                    if torch.any(torch.isnan(loss)):
                        self.encounter_nan = True
                        raise RuntimeError("Encountering nan, stop training")

                    self.optimizer_instance.zero_grad()
                    loss.backward()
                    return loss

                if self.config.optimizer == "adam_LBFGS":
                    if iter > self.config.optimize_next_iterations:
                        self.config.optimizer = "adam_LBFGS_next"
                        self.__initialize_optimizer()

                self.optimizer_instance.step(closure)
                loss = loss_temp_list[-1]

            # Evaluate ---------------------------------------------------------------
            if iter % self.config.print_every == 0 or iter == self.config.iterations:
                for batch_data in self.val_loader:
                    self.model.eval()
                    val_loss = self.model.evaluate(
                        data=batch_data,
                        output_dir=self.evaluate_output_dir,
                        current_iterations=iter,
                    )

            # Record ------------------------------------------------
            loss_history.append([
                iter,
                loss.item(),
                val_loss.item(),
                self.optimizer_instance.param_groups[0]["lr"],
            ])
            postfix = {
                "Train_loss": "{:.3e}".format(loss.item()),
                "Val_loss": "{:.3e}".format(val_loss.item()),
                "lr": self.optimizer_instance.param_groups[0]["lr"],
            }
            pbar.set_postfix(postfix)

            # Save ---------------------------------------------------------------
            model_path = os.path.join(self.model_output_dir, "temp_training_model_{}.pkl".format(iter))
            torch.save(self.model.state_dict(), model_path)

            # LR step ---------------------------------------------------------------
            if self.scheduler_instance is not None:
                self.scheduler_instance.step()

        self.results["loss_history"] = np.array(loss_history)

    def restore(self):
        """
        Restore the best model based on the minimum loss from the loss history.

        Returns:
            torch.nn.Module: The best model.
        Raises:
            RuntimeError: If loss_history is None is False.
        """

        loss_history = self.results.get("loss_history")
        if loss_history is None:
            raise RuntimeError("Cannot restore without loss history. Make sure the 'results' attribute has 'loss_history'.")

        # Find the model with the smallest loss
        best_loss_index = loss_history[:, 1].argmin()
        iteration = int(loss_history[best_loss_index, 0])
        loss_train = loss_history[best_loss_index, 1]
        loss_test = loss_history[best_loss_index, 2]

        # load model
        filepath = os.path.join(self.model_output_dir, "temp_training_model_{}.pkl".format(iteration))
        self.best_model = torch.load(filepath)

        # save relevant information
        contents = ("\n" + "Train completion time: " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + "\n" +
                    "Task name: {}".format(self.config.taskname) + "\n" + "Model name: {}".format(self.model.__class__.__name__) + "\n" +
                    "Best model at iteration: {}".format(iteration) + "\n" + "Train loss: {:.3e}".format(loss_train) + "\n" +
                    "Test loss: {:.3e}".format(loss_test))
        self.logger.info(contents)

    def output_results(self, info):
        ####################################################################################
        # Save best model
        model_filename = f"train-model.pkl"
        model_path = os.path.join(self.result_dir, model_filename)
        torch.save(self.best_model, model_path)

        ####################################################################################
        # Save loss history (for txt and png)
        loss_filename = f"train-loss.txt"
        loss_path = os.path.join(self.result_dir, loss_filename)
        np.savetxt(loss_path, self.results["loss_history"])

        fig_filename = f"train-loss.png"
        fig_path = os.path.join(self.result_dir, fig_filename)

        def save_loss_history_figure(loss_history, fig_path):
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.semilogy(loss_history[:, 0], loss_history[:, 1], "b", label="train loss")
            ax1.semilogy(loss_history[:, 0], loss_history[:, 2], "g", label="test loss")
            ax1.legend(loc=1)
            ax1.set_ylabel("loss")

            ax2 = ax1.twinx()
            ax2.semilogy(loss_history[:, 0], loss_history[:, 3], "r", label="lr")
            ax2.set_ylabel("lr")
            ax2.set_xlabel("EPOCHS")
            ax2.legend(loc=2)

            best_loss_index = self.results["loss_history"][:, 1].argmin()
            iteration = int(self.results["loss_history"][best_loss_index, 0])
            plt.axvline(
                x=iteration,
                color="r",
                linestyle="--",
                label="saved model iteration",
            )

            plt.tight_layout()
            plt.savefig(fig_path, format="png")

        save_loss_history_figure(self.results["loss_history"], fig_path)

        ####################################################################################
        # Save train info
        if info is not None:
            info.pop("data")
            info.pop("logger")
            self.logger.info(info)

        # Logs all the attributes and their values present in the given config object.
        if self.config is not None:
            keys_values_pairs = []  # List to store attribute-name and attribute-value pairs
            for attr_name in dir(self.config):
                if not attr_name.startswith("__"):  # Exclude private attributes
                    attr_value = getattr(self.config, attr_name)  # Get the attribute value
                    keys_values_pairs.append("{}: {}".format(attr_name, attr_value))  # Store the pair
            # Join the attribute-name and attribute-value pairs with newline separator
            full_output = "\n".join(keys_values_pairs)
            # Log the config values
            self.logger.info("Config values:\n%s", full_output)
