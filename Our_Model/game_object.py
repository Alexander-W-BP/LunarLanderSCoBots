# game object interface
from abc import ABC, abstractmethod
import math

class GameObjectInterface(ABC):
    
    @property
    @abstractmethod
    def category(self):
        pass

    @property
    @abstractmethod
    def xy(self):
        pass

    @xy.setter
    @abstractmethod
    def xy(self, xy):
        pass

    @property
    @abstractmethod
    def orientation(self):
        pass

    @orientation.setter
    @abstractmethod
    def orientation(self, o):
        pass

    @property
    @abstractmethod
    def h_coords(self):
        pass

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def h(self):
        pass

    @property
    @abstractmethod
    def rgb(self):
        pass
    
    @property
    @abstractmethod
    def number(self):
        pass
    
    @number.setter
    @abstractmethod
    def number(self, number):
        pass

    #@property
    #@abstractmethod
    #def visible(self):
    #    pass

    # default behaviour
    @property
    def name(self):
        return str(self.category) + str(self.number)
    
    def distance(self, game_object):
        return math.sqrt((self.xy[0] - game_object.xy[0])**2 + (self.xy[1] - game_object.xy[1])**2)

    def x_distance(self, game_object):
        return self.xy[0] - game_object.xy[0]

    def y_distance(self, game_object):
        return self.xy[1] - game_object.xy[1]

    def __repr__(self):
        return f"{self.name} at ({self.xy[0]}, {self.xy[1]})"
    
    # @property
    # @abstractmethod
    # def dx(self):
    #     pass

    # @property
    # @abstractmethod
    # def dy(self):
    #     pass

    # @property
    # @abstractmethod
    # def xywh(self):
    #     pass

    # @property
    # @abstractmethod
    # def x(self):
    #     pass

    # @property
    # @abstractmethod
    # def y(self):
    #     pass


class LunarLanderObject(GameObjectInterface):
    def __init__(self, name, position):
        self._name = name
        self._position = position

    @property
    def name(self):
        return self._name

    @property
    def xy(self):
        # In LunarLander verwenden wir die Position als Ersatz für xy
        return self._position

    @property
    def h_coords(self):
        # Für LunarLander keine Historie, daher Dummy-Werte oder Position zurückgeben
        return (self._position, self._position)

    @property
    def rgb(self):
        # Dummy-Wert für Farben, falls nicht relevant
        return (255, 255, 255)

    @property
    def orientation(self):
        # Standardwert für Orientation, wenn keine verfügbar ist
        return 0,

    @property
    def w(self):
        # Breite als Dummy, da sie nicht benötigt wird
        return 1

    @property
    def h(self):
        # Höhe als Dummy, da sie nicht benötigt wird
        return 1

    @property
    def xywh(self):
        # Kombiniere xy, Breite und Höhe
        return (*self.xy, self.w, self.h)

    @property
    def category(self):
        # Platzhalterwert, da LunarLander kein direktes Kategorie-Attribut hat
        return "LunarLanderObject"

    @property
    def number(self):
        # Einfache Implementierung: Rückgabe von 1 (oder verwende eine eindeutige ID)
        return 1

    @number.setter
    def number(self, value):
        # Falls `number` gesetzt werden muss, kannst du dies hier speichern
        pass