import os


MAIN_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIRECTORY = os.path.join(MAIN_DIRECTORY, 'simulations')
INPUT_DIRECTORY = os.path.join(MAIN_DIRECTORY, 'inputs')
GSO_DIRECTORY = os.path.join(INPUT_DIRECTORY, 'GSO_Base_Network')
LINE_OUTAGE_FILE = 'line_outage_consequences.csv'
LINE_OUTAGE_PATH = os.path.join(INPUT_DIRECTORY, LINE_OUTAGE_FILE)
AFFECTED_BUSES_FILE = 'affected_buses.csv'
AFFECTED_BUSES_PATH = os.path.join(INPUT_DIRECTORY, AFFECTED_BUSES_FILE)
METADATA_FILE = 'metadata.csv'
METADATA_PATH = os.path.join(RESULTS_DIRECTORY, METADATA_FILE)
