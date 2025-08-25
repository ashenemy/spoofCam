# __init__.py for processors
# import processors so pipeline can discover by name
from .simple_race_swap import Processor as RaceProcessor
from .gender_refiner import Processor as GenderProcessor
from .card_replacer import Processor as CardProcessor

REGISTERED_PROCESSORS = {
    "simple_race_swap": RaceProcessor,
    "gender_refiner": GenderProcessor,
    "card_replacer": CardProcessor
}