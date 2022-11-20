import json
import logging
import os
import weakref
from pathlib import Path


class LossLogger:

    def __init__(self, log_file: str = None, save_dir='log'):
        self.selfInit(log_file, save_dir)
        self.losses = []

        self.logF(log_file)

    def logF(self, log_file):
        if log_file:
            self.load(log_file)

    def save(self, file_name: str):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.saveFile(file_name)

    def saveFile(self, file_name):
        with open(self.save_dir / f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.losses, f)

    def selfInit(self, log_file, save_dir):
        self.log_file = log_file
        self.save_dir = Path(save_dir)

    def record(self, loss: float):
        self.losses.append(loss)

    def load(self, file_path: str):
        self.fileError(file_path)
        try:
            self.openFile(file_path)
        except:
            raise Exception("File Error")

    def openFile(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            self.losses = json.load(f)

    def fileError(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError("File Path Error")


loggerNameList = weakref.WeakValueDictionary()


def loggerSave(cls):
    def wrapper(name, *args, **kwargs):
        instance = loggerNameList[name]
        if name not in loggerNameList:
            instance = cls(name, *args, **kwargs)
            loggerNameList[name] = instance

        return instance
    return wrapper


@loggerSave
class Logger:
    log_folder = Path('log')

    def __init__(self, fileName: str):
        self.selfLogFile(fileName)
        self.selfLog(fileName)
        self.selfSetLevel()

        fmt = self.selfSetFormat()
        self.setHandler(fmt)
        self.addHandler()

    def addHandler(self):
        self._logger.addHandler(self._console_handler)
        self._logger.addHandler(self._file_handler)

    def selfSetLevel(self):
        self._console_handler.setLevel(logging.DEBUG)
        self._file_handler.setLevel(logging.DEBUG)

    def selfHandler(self):
        self._console_handler = logging.StreamHandler()
        self._file_handler = logging.FileHandler(self._log_file, encoding='utf-8')

    def setHandler(self, fmt):
        self._console_handler.setFormatter(fmt)
        self._file_handler.setFormatter(fmt)

    def selfLogFile(self, fileName):
        self.log_folder.mkdir(exist_ok=True, parents=True)
        self._log_file = self.log_folder / (fileName + '.log')

    def error(self, msg, exc_info=False):
        self._logger.error(msg, exc_info=exc_info)

    def selfSetFormat(self):
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        return fmt

    def selfLog(self, fileName):
        self.selfHandler()
        self._logger = logging.getLogger(fileName)
        self._logger.setLevel(logging.DEBUG)
