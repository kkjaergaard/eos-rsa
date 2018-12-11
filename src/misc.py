import logging


class Signal:
    """
    Signal class for observer-subscriber pattern, largely inspired (read: copied) by https://blog.abstractfactory.io/dynamic-signals-in-pyqt/
    """

    def __init__(self):
        self.__subscribers = []
        self.logger = logging.getLogger(__name__)

    def emit(self, *args, **kwargs):
        for subs in self.__subscribers:
            subs(*args, **kwargs)

    def connect(self, func):
        self.__subscribers.append(func)

    def disconnect(self, func):
        try:
            self.__subscribers.remove(func)
        except ValueError:
            self.logger.warning("Function {} not removed from signal {}".format(func, self))
