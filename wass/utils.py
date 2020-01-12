"""utils.py

The file prevides helper methods and classes for the entire repository:
    - Training History
"""
import os

from typing import Tuple, Dict, List, Union


class TrainingHistory:
    """Training History
        
    Attributes:
        path {str} -- path where to save the experiment
        exp_name {str} -- experiment name (will be folder name within path)
        data {Dict[str, List[Union[float, bool]]]} -- data from experiment
        dir {str} -- path to experiment directory
    """

    def __init__(
        self: "TrainingHistory",
        path: str,
        exp_name: str,
        data: Dict[str, List[Union[float, bool]]] = None,
    ) -> None:
        """Initialization
        
        Arguments:
            path {str} -- path where to save the experiment
            exp_name {str} -- experiment name (will be folder name within path)
        
        Keyword Arguments:
            data {Dict[str, List[Union[float, bool]]]} -- data from previews 
                experiment (default: {None})
        """
        self.path = path
        self.exp_name = exp_name

        self.data = (
            {"training_loss": [], "validation_loss": [], "halved": []}
            if data is None
            else data
        )

        self.dir = os.path.join(path, exp_name)
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)

    def __len__(self: "TrainingHistory") -> int:
        """Length

        Returns:
            int -- number of epoch passed
        """
        return len(self.data["training_loss"])

    def __iadd__(
        self: "TrainingHistory", datum: Tuple[float, float, bool]
    ) -> "TrainingHistory":
        """Incremental Addition
        
        Arguments:
            datum {Tuple[float, float, bool]} -- snapshot of the tr, cv loss, 
                and halved (for the learning rate)

        Returns:
            TrainingHistory -- modified training history
        """
        tr_loss, cv_loss, halved = datum
        self.data["training_loss"].append(tr_loss)
        self.data["validation_loss"].append(cv_loss)
        self.data["halved"].append(halved)

        return self

    def save(self: "TrainingHistory") -> None:
        """Save to CSV Format
        """
        keys = ";".join(self.data.keys())
        data = "\n".join(
            (
                ";".join((str(self.data[key][i]) for key in self.data.keys()))
                for i in range(len(self))
            )
        )

        file_path = os.path.join(self.dir, "history.csv")
        with open(file_path, "w") as f:
            f.write(f"{keys}\n{data}")

    @classmethod
    def load(cls: "TrainingHistory", path: str) -> "TrainingHistory":
        """Load from CSV File
        
        Arguments:
            path {str} -- path to the training history csv file

        Returns:
            TrainingHistory -- loaded training history
        """
        with open(path, "r") as f:
            lines = f.readlines()

        path, exp_name, _ = path.split("/")[-4:]
        keys = [key.strip() for key in lines[0].split(";")]

        def convert(x: str) -> Union[float, bool]:
            if "False" in x or "True" in x:
                return bool(x)
            return float(x)

        data = {
            key: [
                convert(lines[1 + i].split(";")[k].strip())
                for i in range(len(lines) - 1)
            ]
            for k, key in enumerate(keys)
        }

        history = cls(path, exp_name, data=data)

        return history
